from models.micro_grid_models import DewhModel, GridModel, PvModel, ResDemandModel
from models.agents import MpcAgent

from typing import List as List_T

from models.mld_model import MldInfo, MldModel

from utils.func_utils import ParNotSet

import cvxpy as cvx
from abc import abstractmethod, ABC

import pandas as pd

from datetime import datetime as DateTime, timedelta as TimeDelta

import numpy as np
from utils.matrix_utils import atleast_2d_col
from structdict import StructDictMixin, struct_prop_dict

from controllers.mpc_controller.mpc_controller import MpcController, MpcBuildRequiredError


class MldSimLog(StructDictMixin, dict):
    _internal_names = ['_nan_insert']

    LogEntry_k = struct_prop_dict('LogEntry_k')

    def __init__(self):
        super(MldSimLog, self).__init__()
        self._reset()

    def _reset(self):
        self._base_dict_clear()
        self._nan_insert = {}

    @staticmethod
    def _nan_if_num(var):
        if np.issubsctype(var, np.number) or np.issubsctype(var, np.bool_):
            return var * np.NaN
        else:
            return atleast_2d_col([None] * var.size)

    def set_sim_k(self, k, sim_k=None, **kwargs):
        try:
            del self[k]
        except KeyError:
            pass
        self.update_sim_k(k=k, sim_k=sim_k, **kwargs)

    def update_sim_k(self, k, sim_k=None, **kwargs):
        sim_k = sim_k if sim_k is not None else {}
        sim_k.update(kwargs)

        insert = self.LogEntry_k(self._nan_insert)
        if self.get(k):
            insert.update(self[k])

        for var_name, var_k in sim_k.items():
            if var_k is not None:
                var_k = atleast_2d_col(var_k)
                insert_var = insert.get(var_name)
                if insert_var is not None and insert_var.shape != var_k.shape:
                    raise ValueError("shape of var_k must match previous inserts")
                else:
                    insert[var_name] = var_k

        self[k] = insert

        if set(insert).difference(self._nan_insert):
            self._nan_insert = {var_name: self._nan_if_num(var) for var_name, var in insert.items()}

    def get_concat_log(self):
        nan_insert = self._nan_insert
        for k, log_entry_k in self.items():
            if len(log_entry_k) != len(nan_insert):
                insert = self.LogEntry_k(nan_insert)
                insert.update(log_entry_k)
                self[k] = insert

        index = sorted(self)

        var_lists = {var_name: [] for var_name in nan_insert}
        for k in index:
            for var_name in var_lists:
                var_lists[var_name].append(self[k][var_name])

        dfs = {}
        for var_name, var_list in var_lists.items():
            var_seq = np.array(var_list)
            if var_seq.size:
                var_seq = var_seq.squeeze(axis=2)
                dfs[var_name] = pd.DataFrame(var_seq)

        df = pd.concat(dfs, keys=list(dfs), axis=1)
        df.columns.names = ['var_names', 'var_index']
        df.index = index
        df.index.name = 'k'

        return df


class MicroGridAgentBase(MpcAgent, ABC):
    @abstractmethod
    def __init__(self, device_type=None, device_id=None,
                 sim_model=ParNotSet, control_model=ParNotSet,
                 N_p=None, N_tilde=None, omega_profile=None, profile_t0=DateTime(2018, 12, 3),
                 omega_scenarios_profile=None, scenarios_t0=pd.Timedelta(0), forecast_lag='1D'):
        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=sim_model, control_model=control_model,
                         N_p=N_p, N_tilde=N_tilde)

        self.omega_profile = None
        self.profile_t0 = profile_t0
        self.set_omega_profile(omega_profile=omega_profile, profile_t0=profile_t0)

        self.omega_scenarios = None
        self._omega_scenario_values = None
        self.intervals_per_day = None
        self.num_scenarios = None
        self.scenarios_t0 = scenarios_t0
        self.set_omega_scenarios(omega_scenarios_profile=omega_scenarios_profile, scenarios_t0=scenarios_t0)

        self.forecast_lag = forecast_lag
        self.sim_log = MldSimLog()

        self._variables_k_act = None

    @property
    def variables_k_opt(self):
        return self._mpc_controller.variables_k

    @property
    def variables_k_act(self):
        return self._variables_k_act

    def set_omega_scenarios(self, omega_scenarios_profile=ParNotSet, scenarios_t0=None):
        scenarios_t0 = scenarios_t0 or self.scenarios_t0
        omega_scenarios = omega_scenarios_profile
        if omega_scenarios_profile is not ParNotSet and omega_scenarios_profile is not None:
            omega_scenarios_profile = pd.DataFrame(omega_scenarios_profile)
            n, m = omega_scenarios_profile.shape

            if m != self.mld_info.nomega:
                raise ValueError(
                    f"omega_scenarios_profile must have column dimension of 'nomega':{self.mld_info.nomega} not {m}.")

            self.intervals_per_day = intervals_per_day = pd.Timedelta('1D') // pd.Timedelta(seconds=self.mld_info.dt)
            self.num_scenarios = num_scenarios = n // intervals_per_day

            omega_scenarios_profile.index = pd.timedelta_range(scenarios_t0, periods=n,
                                                               freq=pd.Timedelta(seconds=self.mld_info.dt))
            omega_scenarios_profile.reindex(pd.timedelta_range(scenarios_t0, periods=num_scenarios * intervals_per_day,
                                                               freq=pd.Timedelta(seconds=self.mld_info.dt)))
            omega_scenarios_profile.index.name = 'TimeDelta'

            omega_scenarios = omega_scenarios_profile.stack().values.reshape(intervals_per_day * m, -1, order='F')

            multi_index = pd.MultiIndex.from_product(
                [pd.timedelta_range(scenarios_t0, periods=intervals_per_day,
                                    freq=pd.Timedelta(seconds=self.mld_info.dt)),
                 [f'omega_{index}' for index in range(m)]])

            omega_scenarios = pd.DataFrame(omega_scenarios, index=multi_index)

        self.omega_scenarios = (omega_scenarios if omega_scenarios is not ParNotSet else
                                self.omega_scenarios)

        self._omega_scenario_values = self.omega_scenarios.values if isinstance(omega_scenarios, pd.DataFrame) else None

    def set_omega_profile(self, omega_profile=ParNotSet, profile_t0=None):
        profile_t0 = profile_t0 or self.profile_t0
        if omega_profile is not None and omega_profile is not ParNotSet:
            omega_profile = pd.DataFrame(omega_profile)
            n, m = omega_profile.shape

            if m != self.mld_info.nomega:
                raise ValueError(
                    f"Omega profile must have column dimension of 'nomega':{self.mld_info.nomega} not {m}.")

            omega_profile.columns = [f'omega_{index}' for index in range(m)]
            omega_profile.index = pd.date_range(profile_t0, periods=n, freq=pd.Timedelta(seconds=self.mld_info.dt))
            omega_profile.index.name = 'DateTime'

        self.omega_profile = omega_profile if omega_profile is not ParNotSet else self.omega_profile

    def get_omega_tilde_scenario(self, k, N_tilde=None, num_scenarios=1):
        omega_tilde_scenario = None
        N_tilde = N_tilde if N_tilde is not None else self.N_tilde
        if self.omega_scenarios is not None:
            scenarios = self._omega_scenario_values
            nomega = self.mld_info.nomega
            row = (k % self.intervals_per_day) * nomega

            assert (scenarios.flags.f_contiguous)

            limit = scenarios.size - row - (N_tilde * nomega) - 1

            if limit <= 0 or limit < (N_tilde * nomega * num_scenarios):
                raise ValueError("Insufficient number of scenarios to draw from.")

            valid_columns = int(
                np.unravel_index(indices=limit, dims=scenarios.shape, order='F')[1]) - 1

            omega_tilde_scenarios = []
            for column_sel in np.random.randint(low=0, high=valid_columns, size=num_scenarios):
                # flat_index = np.ravel_multi_index(multi_index=[row, column_sel], dims=scenarios.shape, order='F')
                flat_index = scenarios.shape[0] * column_sel + row
                omega_tilde_scenarios.append(
                    atleast_2d_col(scenarios.ravel(order='F')[flat_index:flat_index + (N_tilde * nomega)]))
            if omega_tilde_scenarios:
                omega_tilde_scenario = np.hstack(omega_tilde_scenarios)

        return omega_tilde_scenario

    def get_omega_tilde_k_act(self, k, N_tilde=1):
        if self.omega_profile is not None:
            df = self.omega_profile
            start = int(pd.Timedelta(self.forecast_lag) / df.index.freq) + k
            end = start + N_tilde
            return atleast_2d_col(df.values[start:end].flatten(order='C'))
        else:
            return None

    def get_omega_tilde_k_hat(self, k, N_tilde=1, deterministic=False):
        if deterministic:
            return self.get_omega_tilde_k_act(k=k, N_tilde=N_tilde)
        elif self.omega_profile is not None:
            df = self.omega_profile
            start = k
            end = start + N_tilde
            return atleast_2d_col(df.values[start:end].flatten(order='C'))
        else:
            return None

    def update_omega_tilde_k(self, k, deterministic=False):
        self._mpc_controller.omega_tilde_k = self.get_omega_tilde_k_hat(k=k, N_tilde=self.N_tilde,
                                                                        deterministic=deterministic)

    def sim_k(self, k, omega_k=None, solver=None):
        var_k = self.variables_k_opt
        omega_k = omega_k if omega_k is not None else self.get_omega_tilde_k_act(k=k, N_tilde=1)
        lsim_k = self.sim_model.mld_numeric.lsim_k(x_k=var_k.x,
                                                   u_k=var_k.u,
                                                   omega_k=omega_k,
                                                   mu_k=None, solver=solver)

        self._variables_k_act = lsim_k.copy()
        lsim_k.omega_hat = var_k.omega
        return lsim_k

    def sim_step_k(self, k, omega_k=None, solver=None):
        if self.sim_log is None:
            self.sim_log = MldSimLog()

        lsim_k = self.sim_k(k=k, omega_k=omega_k, solver=solver)
        self.x_k = lsim_k.x_k1
        self.mpc_controller.variables_k_neg1 = lsim_k
        self.sim_log.set_sim_k(k=k, sim_k=lsim_k)
        return lsim_k

    @property
    @abstractmethod
    def power_N_tilde(self):
        pass

    @property
    @abstractmethod
    def power_k_act(self):
        pass

    @property
    @abstractmethod
    def objective(self):
        return self._mpc_controller.objective

    @property
    @abstractmethod
    def constraints(self):
        return self._mpc_controller.constraints


class MicroGridDeviceAgent(MicroGridAgentBase):

    def set_device_objective_atoms(self, objective_weights_struct=None, **kwargs):
        self._mpc_controller.set_std_obj_atoms(objective_atoms_struct=objective_weights_struct, **kwargs)

    def build_device(self, with_std_objective=True, with_std_constraints=True, sense=None,
                     disable_soft_constraints=False):
        self._mpc_controller.build(with_std_objective=with_std_objective, with_std_constraints=with_std_constraints,
                                   sense=sense,
                                   disable_soft_constraints=disable_soft_constraints)


class DewhAgentMpc(MicroGridDeviceAgent):

    def __init__(self, device_type='dewh', device_id=None,
                 sim_model=ParNotSet, control_model=ParNotSet,
                 N_p=None, N_tilde=None, omega_profile=None, profile_t0=DateTime(2018, 12, 3),
                 omega_scenarios_profile=None, scenarios_t0=pd.Timedelta(0), forecast_lag='1D'):
        sim_model = sim_model if sim_model is not ParNotSet else DewhModel(const_heat=False)
        control_model = control_model if control_model is not ParNotSet else DewhModel(const_heat=True)

        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=sim_model, control_model=control_model,
                         N_p=N_p, N_tilde=N_tilde, omega_profile=omega_profile, profile_t0=profile_t0,
                         omega_scenarios_profile=omega_scenarios_profile, scenarios_t0=scenarios_t0,
                         forecast_lag=forecast_lag)

    def sim_k(self, k, omega_k=None, solver=None):
        omega_k = omega_k if omega_k is not None else self.get_omega_tilde_k_act(k=k, N_tilde=1)
        self.sim_model.update_param_struct(D_h=np.asscalar(omega_k))
        return super(DewhAgentMpc, self).sim_k(k=k, omega_k=omega_k, solver=solver)

    @property
    def power_N_tilde(self):
        return self._mpc_controller.variables.u.var_N_tilde * self.control_model.param_struct.P_h_Nom

    @property
    def power_k_act(self):
        return self.variables_k_act.u * self.sim_model.param_struct.P_h_Nom

    @property
    def objective(self):
        return self._mpc_controller.objective

    @property
    def constraints(self):
        return self._mpc_controller.constraints


class PvAgentMpc(MicroGridDeviceAgent):
    def __init__(self, device_type='pv', device_id=None,
                 sim_model=ParNotSet, control_model=ParNotSet,
                 N_p=None, N_tilde=None, omega_profile=None, profile_t0=DateTime(2018, 12, 3),
                 omega_scenarios_profile=None, scenarios_t0=pd.Timedelta(0), forecast_lag='1D'):
        sim_model = sim_model if sim_model is not ParNotSet else PvModel()
        control_model = control_model if control_model is not ParNotSet else PvModel()

        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=sim_model, control_model=control_model,
                         N_p=N_p, N_tilde=N_tilde, omega_profile=omega_profile, profile_t0=profile_t0,
                         omega_scenarios_profile=omega_scenarios_profile, scenarios_t0=scenarios_t0,
                         forecast_lag=forecast_lag)

    @property
    def power_N_tilde(self):
        return self._mpc_controller.variables.y.var_N_tilde

    @property
    def power_k_act(self):
        return self.variables_k_act.y

    @property
    def objective(self):
        return self._mpc_controller.objective

    @property
    def constraints(self):
        return self._mpc_controller.constraints


class ResDemandAgentMpc(MicroGridDeviceAgent):
    def __init__(self, device_type='resd', device_id=None,
                 sim_model=ParNotSet, control_model=ParNotSet,
                 N_p=None, N_tilde=None, omega_profile=None, profile_t0=DateTime(2018, 12, 3),
                 omega_scenarios_profile=None, scenarios_t0=pd.Timedelta(0), forecast_lag='1D'):
        sim_model = sim_model if sim_model is not ParNotSet else ResDemandModel()
        control_model = control_model if control_model is not ParNotSet else ResDemandModel()

        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=sim_model, control_model=control_model,
                         N_p=N_p, N_tilde=N_tilde, omega_profile=omega_profile, profile_t0=profile_t0,
                         omega_scenarios_profile=omega_scenarios_profile, scenarios_t0=scenarios_t0,
                         forecast_lag=forecast_lag)

    @property
    def power_N_tilde(self):
        return self._mpc_controller.variables.y.var_N_tilde

    @property
    def power_k_act(self):
        return self.variables_k_act.y

    @property
    def objective(self):
        return self._mpc_controller.objective

    @property
    def constraints(self):
        return self._mpc_controller.constraints


class GridAgentMpc(MicroGridAgentBase):

    def __init__(self, device_type='grid', device_id=None, N_p=None, N_tilde=None):

        grid_model = GridModel(num_devices=0)
        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=grid_model, control_model=grid_model,
                         N_p=N_p, N_tilde=N_tilde)

        self.devices: List_T[MicroGridDeviceAgent] = []

    def set_omega_profile(self, omega_profile=ParNotSet, profile_t0=DateTime(2018, 12, 1)):
        return NotImplemented

    @property
    def power_N_tilde(self):
        return self._mpc_controller.variables.y.var_N_tilde

    @property
    def power_k_act(self):
        return self.variables_k_act.y

    @property
    def objective(self):
        try:
            return self._mpc_controller.std_obj_atoms.y.Linear_vector.cost
        except AttributeError:
            return None

    @property
    def constraints(self):
        return self._mpc_controller.constraints

    @property
    def grid_device_powers_N_tilde(self):
        powers_N_tilde_mat = cvx.hstack([device.power_N_tilde for device in self.devices]).T
        return cvx.reshape(powers_N_tilde_mat, (powers_N_tilde_mat.size, 1))

    @property
    def grid_device_powers_k_act(self):
        return np.vstack([device.power_k_act for device in self.devices])

    @property
    def grid_device_objectives(self):
        return [device.objective for device in self.devices]

    @property
    def grid_device_constraints(self):
        return [constraint for device in self.devices for constraint in device.constraints]

    def add_device(self, device):
        if isinstance(device, MicroGridDeviceAgent):
            self.devices.append(device)
        else:
            raise TypeError(f"device type must be a subclass of {MicroGridDeviceAgent.__name__}.")


    def build_grid(self, k=0, N_p=ParNotSet, N_tilde=ParNotSet):
        self.update_horizons(N_p=N_p, N_tilde=N_tilde)
        for device in self.devices:
            if device.N_tilde != self.N_tilde:
                raise ValueError(
                    f"All devices in self.devices must have 'N_tilde' equal to 'self.N_tilde':{self.N_tilde}")

        if self.control_model.num_devices != len(self.devices) or self.sim_model.num_devices != len(self.devices):
            grid_model = GridModel(num_devices=len(self.devices))
            self.update_models(sim_model=grid_model, control_model=grid_model)

        for device in self.devices:
            device.build_device()

        self.mpc_controller.set_constraints(other_constraints=self.grid_device_constraints)
        self.mpc_controller.set_objective(other_objectives=self.grid_device_objectives)

        self.update_omega_tilde_k(k=k)
        self._mpc_controller.build()

    def solve_grid_mpc(self, solver=None, verbose=False, warm_start=True, parallel=False,
                       external_solve=None, *, method=None, **kwargs):

        for device in self.devices:
            if device.mpc_controller.build_required:
                raise MpcBuildRequiredError(
                    f'Mpc problem has not been built for device {device.device_type}:{device.device_id}. A full grid '
                    f'rebuild is required.')

        try:
            grid_solution = self._mpc_controller.solve(solver=solver, verbose=verbose, warm_start=warm_start,
                                                       parallel=parallel,
                                                       external_solve=external_solve, method=method, **kwargs)
        except MpcBuildRequiredError:
            raise MpcBuildRequiredError(
                'Mpc problem has not been built for the grid or a rebuild is required.')

        for device in self.devices:
            device.mpc_controller.solve(external_solve=grid_solution)

    def sim_k(self, k, omega_k=None, solver=None):
        if omega_k is None:
            for device in self.devices:
                device.sim_k(k=k, solver=None)
        return super(GridAgentMpc, self).sim_k(k=k, omega_k=omega_k, solver=None)

    def sim_step_k(self, k, omega_k=None, solver=None):
        for device in self.devices:
            device.sim_step_k(k=k, solver=None)
        omega_k = omega_k if omega_k is not None else self.get_omega_tilde_k_act(k=k)

        lsim_k = super(GridAgentMpc, self).sim_step_k(k, omega_k=omega_k)
        cost = self._mpc_controller.std_obj_atoms.z.Linear_vector.weight.weight_N_tilde[0, 0] * lsim_k.z
        self.sim_log.update_sim_k(k=k, sim_k=dict(cost=cost))
        return lsim_k

    def get_omega_tilde_k_act(self, k, N_tilde=1):
        if N_tilde == 1:
            return self.grid_device_powers_k_act
        else:
            raise ValueError(f"For {self.__class__.__name__} objects, omega_tilde_act is only available for N_tilde=1")

    def get_omega_tilde_k_hat(self, k, N_tilde=1, deterministic=True):
        return NotImplemented

    def update_omega_tilde_k(self, k=0, deterministic=False):
        self._mpc_controller.omega_tilde_k = self.grid_device_powers_N_tilde


if __name__ == '__main__':
    import os

    omega_scenarios_profile = pd.read_pickle(os.path.realpath(r'../experiments/data/dewh_omega_profile_df.pickle'))
    dewh_a = DewhAgentMpc(N_p=48)
    dewh_a.set_omega_scenarios(omega_scenarios_profile / dewh_a.mld_info.dt)

    dewh_a.set_device_objective_atoms(q_u=1, q_mu=[10e5, 10e4])

    dewh_a.omega_tilde_k_hat = dewh_a.get_omega_tilde_scenario(0)
    dewh_a.x_k = 50

    dewh_a.mpc_controller.set_constraints(
        other_constraints=[dewh_a.mpc_controller.gen_evo_constraints(omega_tilde_k=dewh_a.get_omega_tilde_scenario(0))
                           for i in range(100)])
