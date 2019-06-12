import contextlib
import functools
import io
from abc import ABC, abstractmethod

import cvxpy as cvx
import numpy as np
import pandas as pd
import wrapt

from controllers.components.mld_evolution_matrices import MldEvoMatrices
from controllers.components.objective_atoms import ObjectiveAtoms
from controllers.components.variables import EvoVariables
from models.mld_model import MldSystemModel, MldModel, MldInfo
from structdict import struct_repr, StructDictMixin, named_struct_dict
from utils.func_utils import ParNotSet
from utils.helper_funcs import eq_all_val
from utils.matrix_utils import atleast_2d_col
from utils.matrix_utils import matmul
from utils.versioning import VersionMixin
from utils.versioning import versioned

import time

class ControllerBuildRequiredError(RuntimeError):
    pass


class ControllerSolverError(RuntimeError):
    pass


def record_overall_time(var_name):
    @wrapt.decorator
    def wrapper(wrapped, self, args, kwargs):
        start = time.time()
        ret = wrapped(*args, **kwargs)
        setattr(self, var_name, time.time()-start)
        return ret
    return wrapper

def build_required_decor(wrapped=None, set=True):
    if wrapped is None:
        return functools.partial(build_required_decor, set=set)

    @wrapt.decorator
    def wrapper(wrapped, self, args, kwargs):
        ret = wrapped(*args, **kwargs)
        if set:
            self.set_build_required()
        else:
            self._build_required = False
        return ret

    return wrapper(wrapped)


class MldSimLog(StructDictMixin, dict):
    _internal_names = ['_nan_insert']

    LogEntry_k = named_struct_dict('LogEntry_k')

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

    def __setitem__(self, k, sim_k):
        self.set_sim_k(k=k, sim_k=sim_k)

    def update_sim_k(self, k, sim_k=None, **kwargs):
        sim_k = sim_k if sim_k is not None else {}

        try:
            sim_k.update(kwargs)
        except AttributeError:
            raise TypeError(f'sim_k must be subtype of dict or None, not: {type(sim_k).__name__!r}')

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

        self._base_dict_setitem(k, insert)

        if set(insert).difference(self._nan_insert):
            self._nan_insert = {var_name: self._nan_if_num(var) for var_name, var in insert.items()}

    update = update_sim_k

    def get_concat_log(self, add_column_levels=None):
        nan_insert = self._nan_insert
        for k, log_entry_k in self.items():
            if len(log_entry_k) != len(nan_insert):
                insert = self.LogEntry_k(nan_insert)
                insert.update(log_entry_k)
                self._base_dict_setitem(k, insert)

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

        df = pd.concat(dfs, keys=list(dfs), axis=1, copy=False)
        df.columns.names = ['var_names', 'var_index']
        df.index = index
        df.index.name = 'k'

        if add_column_levels:
            df = pd.concat([df], keys=[add_column_levels], axis=1, copy=False)

        return df


@versioned(versioned_sub_objects=('model', 'mld_numeric_k', 'mld_numeric_tilde', 'mld_info_k'))
class ControllerBase(ABC, VersionMixin):
    _repr_components = ['mld_numeric_k', 'mld_info_k']

    @build_required_decor
    def __init__(self, model=None, x_k=None, omega_tilde_k=None, N_p=None, N_tilde=None, agent=None,
                 mld_numeric=None, mld_callable=None, mld_symbolic=None, param_struct=None):

        super(ControllerBase, self).__init__()

        self._N_p = N_p if N_p is not None else 0
        self._N_tilde = N_tilde if N_tilde is not None else self._N_p + 1
        self._build_required = True

        if agent is not None and model is not None:
            raise ValueError('agent and model cannot both be set')
        elif agent is not None:
            self._agent = agent
            self._model = None
        else:
            self._model = model if model is not None else MldSystemModel(mld_numeric=mld_numeric,
                                                                         mld_symbolic=mld_symbolic,
                                                                         mld_callable=mld_callable,
                                                                         param_struct=param_struct)
            self._agent = agent

        self._mld_numeric_tilde = None
        self._sim_log = MldSimLog()

        self._x_k = None
        self._omega_tilde_k = None

        self._solve_time_overall = 0
        self._solve_time_solver = 0

        self.reset_components(x_k=x_k, omega_tilde_k=omega_tilde_k)

    def reset_components(self, x_k=None, omega_tilde_k=None):
        mld_info_k = self.mld_info_k
        self._x_k = x_k if x_k is not None else np.zeros((mld_info_k.nx, 1))
        self._omega_tilde_k = omega_tilde_k if omega_tilde_k is not None else np.zeros(
            (mld_info_k.nomega * self.N_tilde, 1))

    def update_horizons(self, N_p=ParNotSet, N_tilde=ParNotSet):
        old_N_p = self.N_p
        old_N_tilde = self.N_tilde

        self._N_p = N_p if N_p is not ParNotSet else self.N_p or 0
        self._N_tilde = N_tilde if N_tilde is not ParNotSet else self._N_p + 1

        if old_N_p != self._N_p or old_N_tilde != self._N_tilde:
            self.reset_components()

    @property
    @abstractmethod
    def variables_k(self):
        raise NotImplementedError("variables_k property needs to be overridden")

    @property
    @abstractmethod
    def variables_k_neg1(self):
        raise NotImplementedError("variables_k_neg1 property needs to be overridden")

    @variables_k_neg1.setter
    @abstractmethod
    def variables_k_neg1(self, variables_k_neg1_struct):
        raise NotImplementedError("variables_k_neg1 property needs to be overridden")

    def get_sim_k(self, k, default=None, x_k=None, u_k=None, omega_k=None):
        if k in self.sim_log:
            return self.sim_log[k]
        elif default is not None:
            return default
        else:
            return self.sim_model.mld_numeric.lsim_k(x_k=x_k, u_k=u_k, omega_k=omega_k)

    @property
    def sim_log(self) -> MldSimLog:
        return self._sim_log

    def sim_step_k(self, k, x_k=None, u_k=None, omega_k=None, mld_numeric_k=None, solver=None, step_state=True):
        var_k = self.variables_k
        omega_k = omega_k if omega_k is not None else var_k.omega
        x_k = x_k if x_k is not None else var_k.x
        u_k = u_k if u_k is not None else var_k.u

        sim_model = mld_numeric_k if mld_numeric_k is not None else self.sim_model.mld_numeric

        lsim_k = sim_model.lsim_k(x_k=x_k,
                                  u_k=u_k,
                                  omega_k=omega_k,
                                  solver=solver)

        if step_state:
            var_k_hat = {var_name + "_hat": var for var_name, var in var_k.items()}
            lsim_k.update(var_k_hat)
            self.sim_log.set_sim_k(k=k, sim_k=lsim_k)
            self.sim_log.update_sim_k(k=k,
                                      time_solve_overall=self._solve_time_overall,
                                      time_in_solver=self._solve_time_solver)
            self.x_k = lsim_k.x_k1
        else:
            del lsim_k.x_k1

        return lsim_k

    @abstractmethod
    @build_required_decor(set=False)
    def build(self, *args, **kwargs):
        self.update_stored_version()

    @abstractmethod
    @record_overall_time(var_name='_solve_time_overall')
    def solve(self, k, x_k=None, omega_tilde_k=None, external_solve=None, *args, **kwargs):
        pass

    @abstractmethod
    def feedback(self, k, x_k=None, omega_tilde_k=None, external_solve=None, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def x_k(self):
        raise NotImplementedError("x_k property needs to be overridden")

    @x_k.setter
    @abstractmethod
    def x_k(self, value):
        raise NotImplementedError("x_k property needs to be overridden")

    @property
    @abstractmethod
    def omega_tilde_k(self):
        raise NotImplementedError("omega_tilde_k property needs to be overridden")

    @omega_tilde_k.setter
    @abstractmethod
    def omega_tilde_k(self, value):
        raise NotImplementedError("omega_tilde_k property needs to be overridden")

    @property
    def N_p(self):
        return self._N_p

    @property
    def N_tilde(self):
        return self._N_tilde

    @property
    def sim_model(self):
        return self._agent.sim_model if self._agent else self._model

    @property
    def control_model(self):
        return self._agent.control_model if self._agent else self._model

    @property
    def mld_numeric_k(self) -> MldModel:
        mld_numeric_tilde = self._mld_numeric_tilde
        return mld_numeric_tilde[0] if mld_numeric_tilde else self.control_model.mld_numeric

    @property
    def mld_numeric_tilde(self):
        return self._mld_numeric_tilde

    @property
    def mld_info_k(self) -> MldInfo:
        return self.mld_numeric_k.mld_info

    @property
    def build_required(self):
        return self._build_required or self.has_updated_version()

    def set_build_required(self):
        self._build_required = True

    def __repr__(self):
        repr_dict = {component: getattr(self, component, ParNotSet) for component in self._repr_components}
        return struct_repr(repr_dict, type_name=self.__class__.__name__, align_values=True, align_padding_width=1,
                           value_format_str='\b{value}')


@versioned(versioned_sub_objects=('variables', 'mld_evo_matrices', 'solve_version'))
class ConstraintSolvedController(ControllerBase):
    _repr_components = ['N_p', 'N_tilde', 'variables', 'mld_numeric_k']

    @build_required_decor
    def reset_components(self, x_k=None, omega_tilde_k=None):
        if self.control_model.mld_numeric is not None:
            self._mld_evo_matrices: MldEvoMatrices = MldEvoMatrices(self)
            self._variables: EvoVariables = EvoVariables(controller=self, x_k=x_k, omega_tilde_k=omega_tilde_k)
            self._std_obj_atoms: ObjectiveAtoms = ObjectiveAtoms(self)
        else:
            self._mld_evo_matrices: MldEvoMatrices = None
            self._variables: EvoVariables = None
            self._std_obj_atoms: ObjectiveAtoms = None

        self._std_evo_constraints = []
        self._other_constraints = []
        self._constraints = []
        self._problem = None

    @property
    def variables(self) -> EvoVariables:
        return self._variables

    @property
    def variables_k(self):
        if self._variables:
            return self._variables.variables_k
        else:
            return EvoVariables.VariablesStruct_k()

    @property
    def variables_k_neg1(self):
        if self._variables:
            return self._variables.variables_k_neg1
        else:
            return EvoVariables.VariablesStruct_k_neg1()

    @variables_k_neg1.setter
    def variables_k_neg1(self, variables_k_neg1_struct):
        if self._variables:
            self._variables.variables_k_neg1 = variables_k_neg1_struct
        else:
            raise AttributeError(f"{self.__class__.__name__} 'variables' attribute has not been initialised.")

    @property
    def mld_evo_matrices(self):
        return self._mld_evo_matrices

    @property
    def constraints(self):
        return self._constraints

    @property
    def problem(self) -> cvx.Problem:
        return self._problem

    @property
    def x_k(self):
        return self._variables.x_k

    @x_k.setter
    def x_k(self, value):
        self._variables.x_k = value

    @property
    def omega_tilde_k(self) -> cvx.Parameter:
        return self._variables.omega.var_N_tilde

    @omega_tilde_k.setter
    def omega_tilde_k(self, value):
        self._variables.omega_tilde_k = value

    def _gen_std_evo_constraints(self):
        std_evo_constraints = self.gen_evo_constraints(self.x_k, self.omega_tilde_k)
        if std_evo_constraints is not None:
            return [std_evo_constraints]
        else:
            return []

    def gen_evo_constraints(self, x_k=None, omega_tilde_k=None, omega_scenarios_k=ParNotSet,
                            N_p=ParNotSet, N_tilde=ParNotSet, mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet,
                            mld_evo_matrices=ParNotSet):
        # H_v @ v_tilde <= H_x @ x_0 + H_omega @ omega_tilde + H_5

        x_k = x_k if x_k is not None else self.x_k
        omega_tilde_k = omega_tilde_k if omega_tilde_k is not None else self.omega_tilde_k

        if mld_evo_matrices is ParNotSet:
            mld_numeric_k = mld_numeric_k if mld_numeric_k is not self.mld_numeric_k else ParNotSet
            mld_numeric_tilde = mld_numeric_tilde if mld_numeric_tilde is not self.mld_numeric_tilde else ParNotSet

            N_p = N_p if N_p is not ParNotSet else self.N_p
            N_tilde = N_tilde if N_tilde is not ParNotSet else self.N_tilde

            if not N_tilde <= self.N_tilde:
                raise ValueError(f"N_tilde: {N_tilde} must be less or equal to self.N_tilde: {self.N_tilde}")

            if not N_p <= self.N_tilde:
                raise ValueError(f"N_p: {N_tilde} must be less or equal to self.N_tilde: {self.N_tilde}")

            if not eq_all_val(mld_numeric_k, mld_numeric_tilde, val=ParNotSet):
                mld_evo_matrices = MldEvoMatrices(self, N_p=N_p, N_tilde=N_tilde,
                                                  mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)
            elif N_tilde == self.N_tilde:
                mld_evo_matrices = self._mld_evo_matrices
            else:
                mld_evo_matrices = self._mld_evo_matrices.get_evo_matrices_N_tilde(N_tilde=N_tilde)

        if mld_evo_matrices is not None and omega_scenarios_k is not None:
            if mld_evo_matrices.mld_info_k.n_constraints:
                if omega_scenarios_k is not ParNotSet:
                    H_omega_omega = np.min(matmul(mld_evo_matrices.constraint['H_omega_N_tilde'], omega_scenarios_k),
                                           axis=1, keepdims=True)
                else:
                    H_omega_omega = matmul(mld_evo_matrices.constraint['H_omega_N_tilde'], omega_tilde_k)

                LHS = (matmul(mld_evo_matrices.constraint['H_v_N_tilde'], self._variables['v']['var_N_tilde']))
                RHS = (matmul(mld_evo_matrices.constraint['H_x_N_tilde'], x_k) + H_omega_omega
                       + mld_evo_matrices.constraint['H_5_N_tilde'])

                return LHS <= RHS
            else:
                return cvx.Constant(0) <= cvx.Constant(0)
        else:
            return None

    @build_required_decor
    def set_constraints(self, std_evo_constaints=ParNotSet, other_constraints=ParNotSet,
                        disable_soft_constraints=False):
        if std_evo_constaints is not ParNotSet:
            self._std_evo_constraints = (std_evo_constaints if std_evo_constaints is not None else
                                         self._gen_std_evo_constraints())
        if other_constraints is not ParNotSet:
            self._other_constraints = other_constraints if other_constraints is not None else []

        if disable_soft_constraints and self.mld_info_k.nmu:
            soft_cons = [
                self.variables.mu.var_N_tilde == np.zeros(self.variables.mu.var_N_tilde.shape)
            ]
        else:
            soft_cons = []

        self._constraints = self._std_evo_constraints + self._other_constraints + soft_cons

    @build_required_decor(set=False)
    def build(self, with_std_constraints=True, disable_soft_constraints=True):
        self._mld_evo_matrices.update()

        if with_std_constraints:
            std_evo_constraints = None
        else:
            std_evo_constraints = []

        self.set_constraints(std_evo_constaints=std_evo_constraints, disable_soft_constraints=disable_soft_constraints)
        assert isinstance(self._constraints, list)

        self._problem = cvx.Problem(cvx.Minimize(cvx.Constant(0)), self._constraints)
        self.update_stored_version()

    @record_overall_time(var_name='_solve_time_overall')
    def solve(self, k, x_k=None, omega_tilde_k=None, external_solve=None,
              solver=None, verbose=False, warm_start=True, parallel=False, *args, method=None, **kwargs):

        if x_k is not None:
            self.x_k = x_k
        if omega_tilde_k is not None:
            self.omega_tilde_k = omega_tilde_k

        k_neg1 = k - 1 if k is not None else k
        self.variables_k_neg1 = self.get_sim_k(k=k_neg1)

        if self.build_required:
            raise ControllerBuildRequiredError(
                f"{self.__class__.__name__} problem has not been built or needs to be rebuilt.")

        if external_solve is None:
            try:
                solution = self._problem.solve(solver=solver,
                                               verbose=verbose,
                                               warm_start=warm_start,
                                               parallel=parallel, method=method, **kwargs)
                self._solve_time_solver = self._problem.solver_stats.solve_time
            except cvx.error.SolverError as se:
                self._solve_time_solver = np.NaN
                with io.StringIO() as std_out_redirect:
                    if verbose == False:
                        with contextlib.redirect_stdout(std_out_redirect):
                            try:
                                self._problem.solve(solver=solver,
                                                    warm_start=warm_start,
                                                    verbose=True,
                                                    parallel=parallel, method=method, **kwargs)
                            except:
                                pass
                            std_out_redirect.seek(0)
                            out = std_out_redirect.read()
                    else:
                        out = ""

                raise ControllerSolverError(f"{se.args[0]}\n\n{out}") from se

            if not np.isfinite(solution):
                raise ControllerSolverError(
                    f"solve() failed with objective: '{solution}', and status: {self._problem.status}")
        else:
            self._solve_time_solver = 0
            solution = external_solve

        return solution

    def feedback(self, k, x_k=None, omega_tilde_k=None, external_solve=None,
                 solver=None, verbose=False, warm_start=True, parallel=False, *args, method=None, **kwargs):

        self.solve(k=k, x_k=x_k, omega_tilde_k=omega_tilde_k, external_solve=external_solve,
                   solver=solver, warm_start=warm_start, verbose=verbose, parallel=parallel, method=method, **kwargs)

        return self.variables_k


class PredictiveController(ConstraintSolvedController):
    pass


class NonPredictiveController(ControllerBase):
    pass
