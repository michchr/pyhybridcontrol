from models.micro_grid_models import DewhModel, GridModel, PvModel, ResDemandModel
from models.agents import ControlledAgent

from typing import MutableMapping, AnyStr, Tuple as Tuple_T

from utils.func_utils import ParNotSet

import cvxpy as cvx
import cvxpy.expressions.expression as cvx_e
from abc import abstractmethod, ABC

import pandas as pd

from datetime import datetime as DateTime

import numpy as np
from utils.matrix_utils import atleast_2d_col

from controllers.mpc_controller import MpcController
from controllers.controller_base import (ControllerBase, ControllerBuildRequiredError, ConstraintSolvedController,
                                         MldSimLog)

from structdict import StructDict, named_struct_dict

import itertools


class MicroGridAgentBase(ControlledAgent, ABC):
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

        self._variables_k_act = None

    @property
    def N_tilde(self):
        def N_tilde_getter(controller: ControllerBase):
            return controller.N_tilde

        return self._get_controllers_attribute(getter_func_or_name=N_tilde_getter)

    @property
    def x_k(self):
        def x_getter(controller: ControllerBase):
            return controller.x_k

        return self._get_controllers_attribute(getter_func_or_name=x_getter)

    @x_k.setter
    def x_k(self, value):
        self.set_x_k(x_k_or_struct=value)

    @property
    def omega_tilde_k(self):
        def omega_getter(controller: ControllerBase):
            return controller.omega_tilde_k

        return self._get_controllers_attribute(getter_func_or_name=omega_getter)

    @omega_tilde_k.setter
    def omega_tilde_k(self, value):
        self.set_omega_tilde_k(omega_tilde_k_or_struct=value)

    def set_x_k(self, controller_name=ParNotSet, x_k_or_struct=None):
        def set_controller_x(controller: ControllerBase, x_k):
            controller.x_k = x_k

        self._set_controllers_attribute(setter_func_or_name=set_controller_x, controller_name=controller_name,
                                        attribute_or_struct=x_k_or_struct)

    def set_omega_tilde_k(self, controller_name=ParNotSet, omega_tilde_k_or_struct=None):
        def set_controller_omega(controller: ControllerBase, omega_tilde_k):
            controller.omega_tilde_k = omega_tilde_k

        self._set_controllers_attribute(setter_func_or_name=set_controller_omega, controller_name=controller_name,
                                        attribute_or_struct=omega_tilde_k_or_struct)

    def _set_controllers_attribute(self, setter_func_or_name, controller_name=ParNotSet, attribute_or_struct=None):
        if isinstance(setter_func_or_name, str):
            attribute_name = setter_func_or_name

            def setter_func(controller: ControllerBase, attribute):
                setattr(controller, attribute_name, attribute)
        else:
            setter_func = setter_func_or_name

        if not isinstance(attribute_or_struct, dict):
            attribute_struct = dict.fromkeys(self.controllers, attribute_or_struct)
        else:
            attribute_struct = attribute_or_struct

        if controller_name is not ParNotSet:
            setter_func(self.controllers[controller_name], attribute_struct[controller_name])
        else:
            for cname, controller in self.controllers.items():
                setter_func(controller, attribute_struct[cname])

    ControllersAttrStruct = named_struct_dict('ControllersAttrStruct')

    def _get_controllers_attribute(self, getter_func_or_name, controller_name=ParNotSet):
        if isinstance(getter_func_or_name, str):
            attribute_name = getter_func_or_name

            def getter_func(controller: ControllerBase):
                getattr(controller, attribute_name, ParNotSet)
        else:
            getter_func = getter_func_or_name

        struct = self.ControllersAttrStruct()
        if controller_name is not ParNotSet:
            struct[controller_name] = getter_func(self.controllers[controller_name])
        else:
            struct.update({cname: getter_func(controller) for cname, controller in self.controllers.items()})

        return struct

    @property
    def sim_logs(self) -> MutableMapping[AnyStr, MldSimLog]:
        return StructDict(
            {controller_name: controller.sim_log for controller_name, controller in self._controllers.items()})

    @property
    def sim_dataframe(self):
        dfs = {}
        for cname, controller in self.controllers.items():
            df = pd.concat([controller.sim_log.get_concat_log()], keys=[(self.device_type, self.device_id, cname)],
                           names=['device_type', 'device_id', 'controller'], axis=1, copy=False)
            dfs[cname] = df

        return pd.concat(dfs.values(), axis=1)

    def get_variables_k_act(self, k):
        return StructDict(
            {controller_name: controller.sim_log.get(k) for controller_name, controller in self._controllers})

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

    def get_omega_tilde_scenario(self, k, N_tilde, num_scenarios=1):
        omega_tilde_scenario = None
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

    OmegaTildeKActStruct = named_struct_dict('OmegaTildeKActStruct')

    def get_omega_tilde_k_act(self, k, controller_name=ParNotSet, N_tilde_or_struct=None):
        def extact_omega_tilde_act(omega_profile, k, N_tilde):
            start = int(pd.Timedelta(self.forecast_lag) / omega_profile.index.freq) + k
            end = start + N_tilde
            return atleast_2d_col(omega_profile.values[start:end].flatten(order='C'))

        if controller_name is not ParNotSet:
            controllers = {controller_name: self.controllers[controller_name]}
        else:
            controllers = self.controllers

        if controller_name is not ParNotSet and N_tilde_or_struct is None:
            N_tilde_struct = {controller_name: controllers[controller_name].N_tilde}
        elif N_tilde_or_struct is None:
            N_tilde_struct = self.N_tilde
        elif isinstance(N_tilde_or_struct, dict):
            N_tilde_struct = N_tilde_or_struct
        else:
            N_tilde_struct = dict.fromkeys(controllers, N_tilde_or_struct)

        struct = self.OmegaTildeKActStruct.fromkeys(controllers, ParNotSet)
        if self.omega_profile is not None:
            for cname, controller in controllers.items():
                struct[cname] = extact_omega_tilde_act(self.omega_profile, k, N_tilde_struct[cname])

        return struct

    OmegaTildeKHatStruct = named_struct_dict('OmegaTildeKHatStruct')

    def get_omega_tilde_k_hat(self, k, controller_name=ParNotSet, N_tilde_or_struct=None,
                              deterministic_or_struct=False):
        def extact_omega_tilde_hat(omega_profile, k, N_tilde):
            start = k
            end = start + N_tilde
            return atleast_2d_col(omega_profile.values[start:end].flatten(order='C'))

        if controller_name is not ParNotSet:
            controllers = {controller_name: self.controllers[controller_name]}
        else:
            controllers = self.controllers

        if isinstance(deterministic_or_struct, dict):
            deterministic_struct = deterministic_or_struct
        else:
            deterministic_struct = dict.fromkeys(controllers, deterministic_or_struct)

        if N_tilde_or_struct is None:
            N_tilde_struct = self.N_tilde
        elif isinstance(N_tilde_or_struct, dict):
            N_tilde_struct = N_tilde_or_struct
        else:
            N_tilde_struct = dict.fromkeys(self.controllers, N_tilde_or_struct)

        struct = self.OmegaTildeKHatStruct.fromkeys(controllers, ParNotSet)
        if self.omega_profile is not None:
            for cname, controller in self.controllers.items():
                if deterministic_struct[cname]:
                    struct[cname] = self.get_omega_tilde_k_act(k=k, controller_name=cname,
                                                               N_tilde_or_struct=N_tilde_struct[cname])[cname]
                else:
                    struct[cname] = extact_omega_tilde_hat(self.omega_profile, k, N_tilde_struct[cname])

        return struct

    def update_omega_tilde_k(self, k, deterministic_or_struct=False):
        omega_tilde_k_hat = self.get_omega_tilde_k_hat(k=k, N_tilde_or_struct=self.N_tilde,
                                                       deterministic_or_struct=deterministic_or_struct)
        for cname, controller in self._controllers.items():
            controller.omega_tilde_k = omega_tilde_k_hat[cname]

    def sim_step_k(self, k, x_k_struct=None, omega_k_struct=None, mld_numeric_k_struct=None, solver=None,
                   step_state=True):

        x_k_struct = x_k_struct if x_k_struct is not None else dict.fromkeys(self.controllers, None)
        omega_k_struct = omega_k_struct if omega_k_struct is not None else (
            self.get_omega_tilde_k_act(k=k, N_tilde_or_struct=1))

        mld_numeric_k_struct = mld_numeric_k_struct if mld_numeric_k_struct is not None else (
            dict.fromkeys(self.controllers, None))

        sims_step_k_struct = StructDict()
        for cname, controller in self._controllers.items():
            sims_step_k_struct[cname] = controller.sim_step_k(k=k, x_k=x_k_struct[cname], omega_k=omega_k_struct[cname],
                                                              mld_numeric_k=mld_numeric_k_struct[cname], solver=solver,
                                                              step_state=step_state)

        return sims_step_k_struct

    Powers_N_Tilde_Struct = named_struct_dict('Powers_N_Tilde_Struct')

    @abstractmethod
    def get_powers_N_tilde(self, k):
        return self.Powers_N_Tilde_Struct()

    Powers_k_Actual_Struct = named_struct_dict('Powers_k_Actual_Struct')

    @abstractmethod
    def get_powers_k_actual(self, k):
        return self.Powers_k_Actual_Struct()

    Objectives_Struct = named_struct_dict('Objectives_Struct')

    def get_objectives(self):
        objectives_struct = self.Objectives_Struct()
        for controller_name, controller in self._controllers.items():
            objectives_struct[controller_name] = getattr(controller, 'objective', None)
        return objectives_struct

    Constraints_Struct = named_struct_dict('Constraints_Struct')

    def get_constraints(self):
        constraints_struct = self.Constraints_Struct()
        for controller_name, controller in self._controllers.items():
            constraints_struct[controller_name] = getattr(controller, 'constraints', None)
        return constraints_struct


class MicroGridDeviceAgent(MicroGridAgentBase):

    def set_device_objective_atoms(self, controller_name, objective_atoms_struct=None, **kwargs):
        controller = self.controllers[controller_name]
        if isinstance(controller, MpcController):
            self.controllers[controller_name].set_std_obj_atoms(objective_atoms_struct=objective_atoms_struct,
                                                                **kwargs)

    def build_device(self, with_std_objective=True, with_std_constraints=True, sense=None,
                     disable_soft_constraints=False):
        for controller_name, controller in self._controllers.items():
            if isinstance(controller, MpcController):
                controller.build(with_std_objective=with_std_objective, with_std_constraints=with_std_constraints,
                                 sense=sense, disable_soft_constraints=disable_soft_constraints)
            else:
                controller.build()


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

    def sim_step_k(self, k, x_k_struct=None, omega_k_struct=None, mld_numeric_k_struct=None, solver=None,
                   step_state=True):
        omega_k_struct = omega_k_struct if omega_k_struct is not None else (
            self.get_omega_tilde_k_act(k=k, N_tilde_or_struct=1))

        x_k_struct = x_k_struct if x_k_struct is not None else self.x_k
        for cname, value in x_k_struct.items():
            if isinstance(value, cvx_e.Expression):
                x_k_struct[cname] = value = value.value
            if value <= self.sim_model.param_struct.T_w:
                x_k_struct[cname] = self.sim_model.param_struct.T_w + 0.1

        mld_numeric_k_struct = mld_numeric_k_struct if mld_numeric_k_struct is not None else (
            {cname: (
                self.sim_model.get_mld_numeric(D_h=np.asscalar(omega_k_struct[cname]),
                                               T_h=np.asscalar(x_k_struct[cname]))) for cname in self.controllers})

        return super(DewhAgentMpc, self).sim_step_k(k=k, x_k_struct=x_k_struct, omega_k_struct=omega_k_struct,
                                                    mld_numeric_k_struct=mld_numeric_k_struct, solver=solver,
                                                    step_state=True)

    def get_powers_N_tilde(self, k):
        powers_N_tilde_struct = self.Powers_N_Tilde_Struct()
        for controller_name, controller in self._controllers.items():
            if isinstance(controller, MpcController):
                powers_N_tilde_struct[controller_name] = (
                        controller.variables.u.var_N_tilde * self.control_model.param_struct.P_h_Nom)
            else:
                powers_N_tilde_struct[controller_name] = (
                        controller.variables_k.u * self.control_model.param_struct.P_h_Nom)

        return powers_N_tilde_struct

    def get_powers_k_actual(self, k):
        powers_k_act_struct = self.Powers_k_Actual_Struct()
        for controller_name, controller in self._controllers.items():
            powers_k_act_struct[controller_name] = controller.sim_log[k].u * self.control_model.param_struct.P_h_Nom

        return powers_k_act_struct


class PvAgentMpc(MicroGridDeviceAgent):
    def __init__(self, device_type='pv', device_id=None,
                 sim_model=ParNotSet, control_model=ParNotSet,
                 N_p=None, N_tilde=None, omega_profile=None, profile_t0=DateTime(2018, 12, 3),
                 omega_scenarios_profile=None, scenarios_t0=pd.Timedelta(0), forecast_lag='1D'):
        sim_model = sim_model if sim_model is not ParNotSet else PvModel()
        control_model = control_model if control_model is not ParNotSet else sim_model

        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=sim_model, control_model=control_model,
                         N_p=N_p, N_tilde=N_tilde, omega_profile=omega_profile, profile_t0=profile_t0,
                         omega_scenarios_profile=omega_scenarios_profile, scenarios_t0=scenarios_t0,
                         forecast_lag=forecast_lag)

    def get_powers_N_tilde(self, k):
        powers_N_tilde_struct = self.Powers_N_Tilde_Struct()
        for controller_name, controller in self._controllers.items():
            if isinstance(controller, ConstraintSolvedController):
                powers_N_tilde_struct[controller_name] = controller.variables.y.var_N_tilde
            else:
                powers_N_tilde_struct[controller_name] = controller.variables_k.y

        return powers_N_tilde_struct

    def get_powers_k_actual(self, k):
        powers_k_act_struct = self.Powers_k_Actual_Struct()
        for controller_name, controller in self._controllers.items():
            powers_k_act_struct[controller_name] = controller.sim_log[k].y

        return powers_k_act_struct


class ResDemandAgentMpc(MicroGridDeviceAgent):
    def __init__(self, device_type='resd', device_id=None,
                 sim_model=ParNotSet, control_model=ParNotSet,
                 N_p=None, N_tilde=None, omega_profile=None, profile_t0=DateTime(2018, 12, 3),
                 omega_scenarios_profile=None, scenarios_t0=pd.Timedelta(0), forecast_lag='1D'):
        sim_model = sim_model if sim_model is not ParNotSet else ResDemandModel()
        control_model = control_model if control_model is not ParNotSet else sim_model

        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=sim_model, control_model=control_model,
                         N_p=N_p, N_tilde=N_tilde, omega_profile=omega_profile, profile_t0=profile_t0,
                         omega_scenarios_profile=omega_scenarios_profile, scenarios_t0=scenarios_t0,
                         forecast_lag=forecast_lag)

    def get_powers_N_tilde(self, k):
        powers_N_tilde_struct = self.Powers_N_Tilde_Struct()
        for controller_name, controller in self._controllers.items():
            if isinstance(controller, ConstraintSolvedController):
                powers_N_tilde_struct[controller_name] = controller.variables.y.var_N_tilde
            else:
                powers_N_tilde_struct[controller_name] = controller.variables_k.y

        return powers_N_tilde_struct

    def get_powers_k_actual(self, k):
        powers_k_act_struct = self.Powers_k_Actual_Struct()
        for controller_name, controller in self._controllers.items():
            powers_k_act_struct[controller_name] = controller.sim_log[k].y

        return powers_k_act_struct


GridDevicesStruct = named_struct_dict('GridDevicesStruct')


class GridAgentMpc(MicroGridAgentBase):

    def __init__(self, device_type='grid', device_id=None,
                 sim_model=ParNotSet, control_model=ParNotSet,
                 N_p=None, N_tilde=None,
                 price_profile=None, profile_t0=DateTime(2018, 12, 3), forecast_lag='1D'):

        sim_model = sim_model if sim_model is not ParNotSet else GridModel(0)
        control_model = control_model if control_model is not ParNotSet else sim_model

        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=sim_model, control_model=control_model, profile_t0=DateTime(2018, 12, 3),
                         N_p=N_p, N_tilde=N_tilde, forecast_lag=forecast_lag)

        self.price_profile = None
        self.set_price_profile(price_profile=price_profile, profile_t0=profile_t0)

        self._devices_struct: MutableMapping[AnyStr, MutableMapping[int, MicroGridDeviceAgent]] = GridDevicesStruct()

        self._devices: Tuple_T[MicroGridDeviceAgent] = ()

    @property
    def grid_sim_dataframe(self):
        dfs = [device.sim_dataframe for device in itertools.chain([self], self.devices)]
        return pd.concat(dfs, axis=1)

    @property
    def devices_struct(self):
        return self._devices_struct

    @property
    def devices(self):
        return self._devices

    @property
    def num_devices(self):
        return len(self._devices)

    @property
    def device_controllers(self):
        chain = itertools.chain.from_iterable((device.controllers.items() for device in self._devices))
        return set(chain)

    def add_device(self, device):
        if isinstance(device, MicroGridDeviceAgent):
            if device.device_type in self._devices_struct:
                self._devices_struct[device.device_type][device.device_id] = device
            else:
                self._devices_struct[device.device_type] = GridDevicesStruct({device.device_id: device})

            devices = []
            for device_type in sorted(self._devices_struct):
                devices_dt = self._devices_struct[device_type]
                for device_id in sorted(devices_dt):
                    devices.append(devices_dt[device_id])
            self._devices = tuple(devices)
        else:
            raise TypeError(f"device type must be a subclass of {MicroGridDeviceAgent.__name__}.")

    def set_omega_profile(self, omega_profile=ParNotSet, profile_t0=DateTime(2018, 12, 1)):
        return NotImplemented

    def set_price_profile(self, price_profile=ParNotSet, profile_t0=None):
        profile_t0 = profile_t0 or self.profile_t0
        if price_profile is not None and price_profile is not ParNotSet:
            price_profile = pd.DataFrame(price_profile)
            n, m = price_profile.shape

            # if m != self.mld_info.nomega:
            #     raise ValueError(
            #         f"Omega profile must have column dimension of 'nomega':{self.mld_info.nomega} not {m}.")

            price_profile.columns = [f'price_{index}' for index in range(m)]
            price_profile.index = pd.date_range(profile_t0, periods=n, freq=pd.Timedelta(seconds=self.mld_info.dt))
            price_profile.index.name = 'DateTime'

        self.price_profile = price_profile if price_profile is not ParNotSet else self.price_profile

    PriceTildeKStruct = named_struct_dict('PriceTildeKStruct')

    def get_price_tilde_k(self, k, controller_name=ParNotSet, N_tilde_or_struct=None):
        def extact_price_tilde_act(price_profile, k, N_tilde):
            start = int(pd.Timedelta(self.forecast_lag) / price_profile.index.freq) + k
            end = start + N_tilde
            return atleast_2d_col(price_profile.values[start:end].flatten(order='C'))

        if controller_name is not ParNotSet:
            controllers = {controller_name: self.controllers[controller_name]}
        else:
            controllers = self.controllers

        if controller_name is not ParNotSet and N_tilde_or_struct is None:
            N_tilde_struct = {controller_name: controllers[controller_name].N_tilde}
        elif N_tilde_or_struct is None:
            N_tilde_struct = self.N_tilde
        elif isinstance(N_tilde_or_struct, dict):
            N_tilde_struct = N_tilde_or_struct
        else:
            N_tilde_struct = dict.fromkeys(controllers, N_tilde_or_struct)

        struct = self.PriceTildeKStruct.fromkeys(controllers, ParNotSet)
        if self.price_profile is not None:
            for cname, controller in controllers.items():
                struct[cname] = extact_price_tilde_act(self.price_profile, k, N_tilde_struct[cname])

        return struct

    def get_powers_N_tilde(self, k):
        powers_N_tilde_struct = self.Powers_N_Tilde_Struct()
        for controller_name, controller in self._controllers.items():
            if isinstance(controller, ConstraintSolvedController):
                powers_N_tilde_struct[controller_name] = controller.variables.y.var_N_tilde
            else:
                powers_N_tilde_struct[controller_name] = controller.variables_k.y

        return powers_N_tilde_struct

    def get_powers_k_actual(self, k):
        powers_k_act_struct = self.Powers_k_Actual_Struct()
        for controller_name, controller in self._controllers.items():
            powers_k_act_struct[controller_name] = controller.sim_log[k].y

        return powers_k_act_struct

    def get_grid_device_powers_N_tilde(self, k):
        powers_N_tilde_struct = self.Powers_N_Tilde_Struct.fromkeys_withfunc(self.controllers, func=lambda k: [])

        device_powers = {}
        for device in self.devices:
            device_powers[(device.device_type, device.device_id)] = device.get_powers_N_tilde(k=k)

        for cname, controller in self.controllers.items():
            for dev_type_id, power_struct in device_powers.items():
                power = power_struct.get(cname)
                if power is None:
                    try:
                        power = power_struct['no_controller']
                    except KeyError:
                        raise ValueError(f'{dev_type_id[0]!r}:{dev_type_id[1]} does not contain controller: {cname!r}')
                powers_N_tilde_struct[cname].append(power[:controller.N_tilde])

        for cname, device_powers in powers_N_tilde_struct.items():
            powers_N_tilde_mat = cvx.hstack(device_powers).T
            powers_N_tilde_struct[cname] = cvx.reshape(powers_N_tilde_mat, (powers_N_tilde_mat.size, 1))

        return powers_N_tilde_struct

    def get_grid_device_powers_k_act(self, k):
        powers_k_act_struct = self.Powers_k_Actual_Struct.fromkeys_withfunc(self.controllers, func=lambda k: [])
        device_powers = {}
        for device in self.devices:
            device_powers[(device.device_type, device.device_id)] = device.get_powers_k_actual(k=k)

        for cname, controller in self.controllers.items():
            for dev_type_id, power_struct in device_powers.items():
                power = power_struct.get(cname)
                if power is None:
                    try:
                        power = power_struct['no_controller']
                    except KeyError:
                        raise ValueError(f'{dev_type_id[0]!r}:{dev_type_id[1]} does not contain controller: {cname!r}')
                powers_k_act_struct[cname].append(power)

        for cname, device_powers in powers_k_act_struct.items():
            powers_k_act_struct[cname] = np.vstack(device_powers)

        return powers_k_act_struct

    def get_grid_device_objectives(self):
        objectives_struct = self.Objectives_Struct.fromkeys_withfunc(self.controllers, func=lambda k: [])
        for device in self.devices:
            device_objectives = device.get_objectives()
            for cname in objectives_struct:
                objective = device_objectives.get(cname)
                if objective is not None:
                    objectives_struct[cname].append(objective)

        return objectives_struct

    def get_grid_device_constraints(self):
        constraints_struct = self.Constraints_Struct.fromkeys_withfunc(self.controllers, func=lambda k: [])
        for device in self.devices:
            device_constraints = device.get_constraints()
            for cname in constraints_struct:
                constraints = device_constraints.get(cname)
                if constraints is not None:
                    constraints_struct[cname].extend(constraints)

        return constraints_struct

    def build_grid(self, k, deterministic_or_struct=True):
        if self.control_model.num_devices != self.num_devices or self.sim_model.num_devices != self.num_devices:
            grid_model = GridModel(num_devices=self.num_devices)
            self.update_models(sim_model=grid_model, control_model=grid_model)

        self.update_omega_tilde_k(k=k, deterministic_or_struct=deterministic_or_struct)

        for device in self.devices:
            device.build_device()

        device_constraints = self.get_grid_device_constraints()
        device_objectives = self.get_grid_device_objectives()
        for cname, controller in self.controllers.items():
            if isinstance(controller, ConstraintSolvedController):
                controller.set_constraints(other_constraints=device_constraints[cname])
            if isinstance(controller, MpcController):
                controller.set_objective(other_objectives=device_objectives[cname])
            controller.build()

    def solve_grid_mpc(self, k, solver=None, verbose=False, warm_start=True, parallel=False,
                       external_solve=None, *, method=None, **kwargs):

        for device in self.devices:
            for cname, controller in device.controllers.items():
                if controller.build_required:
                    raise ControllerBuildRequiredError(
                        f'Controller:{cname!r} has not been built for device {device.device_type}:{device.device_id}.')

        for cname, controller in self.controllers.items():
            try:
                grid_solution = controller.solve(k=k, solver=solver, verbose=verbose, warm_start=warm_start,
                                                 parallel=parallel,
                                                 external_solve=external_solve, method=method, **kwargs)
            except ControllerBuildRequiredError:
                raise ControllerBuildRequiredError(
                    f'Controller:{cname!r} has not been built for the grid or a rebuild is required.')

            for device in self.devices:
                if cname in device.controllers:
                    device.controllers[cname].solve(k=k, external_solve=grid_solution)

    def sim_step_k(self, k, x_k_struct=None, omega_k_struct=None, mld_numeric_k_struct=None, solver=None,
                   step_state=True):
        for device in self.devices:
            device.sim_step_k(k=k, solver=solver, step_state=step_state)

        lsim_k = super(GridAgentMpc, self).sim_step_k(k=k, x_k_struct=x_k_struct, omega_k_struct=omega_k_struct,
                                                      mld_numeric_k_struct=mld_numeric_k_struct, solver=solver,
                                                      step_state=step_state)

        prices_k = self.get_price_tilde_k(k=k, N_tilde_or_struct=1)
        for cname, controller in self.controllers.items():
            sim_k = lsim_k[cname]

            additems = {}
            additems['p_imp'] = sim_k.z
            additems['p_exp'] = sim_k.y - sim_k.z
            additems['cost'] = sim_k.z * prices_k[cname]

            controller.sim_log.update_sim_k(k=k, sim_k=additems)
            sim_k.update(additems)

        return lsim_k

    def update_omega_tilde_k(self, k, deterministic_or_struct=False):
        for device in self.devices:
            device.update_omega_tilde_k(k, deterministic_or_struct=deterministic_or_struct)

        device_powers = self.get_grid_device_powers_N_tilde(k=k)
        for cname, controller in self.controllers.items():
            power = device_powers[cname]
            controller.omega_tilde_k = power

    def get_omega_tilde_k_hat(self, k, controller_name=ParNotSet, N_tilde_or_struct=None,
                              deterministic_or_struct=False):
        raise NotImplementedError

    def get_omega_tilde_k_act(self, k, controller_name=ParNotSet, N_tilde_or_struct=None):
        if N_tilde_or_struct is None:
            N_tilde_struct = self.N_tilde
        elif isinstance(N_tilde_or_struct, dict):
            N_tilde_struct = N_tilde_or_struct
        else:
            N_tilde_struct = dict.fromkeys(self.controllers, N_tilde_or_struct)

        if all((N_tilde == 1 for N_tilde in N_tilde_struct.values())):
            return self.get_grid_device_powers_k_act(k=k)
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    import os
    import pandas as pd
    from controllers.no_controller import NoController

    d1 = DewhAgentMpc()
    d1.add_controller('mpc', MpcController, N_tilde=10, N_p=9)

    g1 = GridAgentMpc()
    g1.add_controller('mpc', MpcController, N_tilde=10, N_p=9)

    g1.add_device(d1)

    omega_pv_profile = pd.read_pickle(
        os.path.realpath(r'../experiments/data/pv_supply_norm_1000w_15min.pickle')) / 1000

    omega_resd_profile = pd.read_pickle(
        os.path.realpath(r'../experiments/data/res_demand_norm_1000w_15min.pickle')) / 1000

    g2 = GridAgentMpc()

    resd1 = ResDemandAgentMpc(omega_profile=omega_resd_profile)
    pv1 = PvAgentMpc(omega_profile=omega_pv_profile)

    g2.add_controller('no_controller', NoController, N_p=9, N_tilde=10)
    resd1.add_controller('no_controller', NoController, N_p=9, N_tilde=10)
    pv1.add_controller('no_controller', NoController, N_p=9, N_tilde=10)

    g2.add_device(pv1)
    g2.add_device(resd1)

    # omega_scenarios_profile = pd.read_pickle(os.path.realpath(r'../experiments/data/dewh_omega_profile_df.pickle'))
    # dewh_a = DewhAgentMpc(N_p=48)
    # dewh_a.set_omega_scenarios(omega_scenarios_profile / dewh_a.mld_info.dt)
    #
    # dewh_a.set_device_objective_atoms(q_u=1, q_mu=[10e5, 10e4])
    #
    # dewh_a.omega_tilde_k_hat = dewh_a.get_omega_tilde_scenario(0)
    # dewh_a.x_k = 50
    #
    # dewh_a.mpc_controller.set_constraints(
    #     other_constraints=[dewh_a.mpc_controller.gen_evo_constraints(omega_tilde_k=dewh_a.get_omega_tilde_scenario(0))
    #                        for i in range(100)])
