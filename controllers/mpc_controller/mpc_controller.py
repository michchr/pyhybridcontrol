import cvxpy as cvx
import numpy as np

import cvxpy.expressions.expression as cvx_e

from controllers.mpc_controller.mpc_components import MpcEvoMatrices, MpcVariables, MpcObjectiveWeights
from controllers.mpc_controller.mpc_utils import process_base_args
from models.mld_model import MldModel, MldInfo, MldSystemModel
from utils.decorator_utils import process_method_args_decor
from utils.matrix_utils import atleast_2d_col, matmul
from structdict import struct_prop_fixed_dict, struct_repr
from utils.helper_funcs import eq_all_val

from utils.func_utils import ParNotSet
from utils.versioning import VersionMixin, versioned, VersionObject

import wrapt
import io
import contextlib

import functools


@versioned(versioned_sub_objects=('model', 'mld_numeric_k', 'mld_numeric_tilde', 'mld_info_k'))
class MpcBase(VersionMixin):
    def __init__(self, model=None, N_p=None, N_tilde=None, agent=None,
                 mld_numeric=None, mld_callable=None, mld_symbolic=None, param_struct=None):

        super(MpcBase, self).__init__()
        self._N_p = N_p if N_p is not None else 0
        self._N_tilde = N_tilde if N_tilde is not None else self._N_p + 1

        if agent is not None:
            self._agent = agent
            self._model = None
        else:
            self._model = model if model is not None else MldSystemModel(mld_numeric=mld_numeric,
                                                                         mld_symbolic=mld_symbolic,
                                                                         mld_callable=mld_callable,
                                                                         param_struct=param_struct)
            self._agent = None

        self._mld_numeric_tilde = None

    @property
    def N_p(self):
        return self._N_p

    @property
    def N_tilde(self):
        return self._N_tilde

    @property
    def model(self):
        return self._agent.control_model if self._agent else self._model

    @property
    def mld_numeric_k(self) -> MldModel:
        mld_numeric_tilde = self._mld_numeric_tilde
        return mld_numeric_tilde[0] if mld_numeric_tilde else self.model.mld_numeric

    @property
    def mld_numeric_tilde(self):
        return self._mld_numeric_tilde

    @property
    def mld_info_k(self) -> MldInfo:
        return self.mld_numeric_k.mld_info


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


@versioned(versioned_sub_objects=('variables', 'sys_evo_matrices', 'solve_version'))
class MpcController(MpcBase):
    _data_types = ['std_obj_weights', 'variables', 'mld_numeric_k']

    @build_required_decor
    def __init__(self, model=None, N_p=None, N_tilde=None, agent=None,
                 mld_numeric=None, mld_callable=None, mld_symbolic=None, param_struct=None,
                 x_k=None, omega_tilde_k=None):

        super(MpcController, self).__init__(model=model, N_p=N_p, N_tilde=N_tilde, agent=agent,
                                            mld_numeric=mld_numeric, mld_callable=mld_callable,
                                            mld_symbolic=mld_symbolic, param_struct=param_struct)

        self._x_k = None
        self._omega_tilde_k = None
        self.solve_version = VersionObject('solve_version')
        self._update_components(x_k=x_k, omega_tilde_k=omega_tilde_k)

    @build_required_decor
    def _update_components(self, x_k=None, omega_tilde_k=None):
        self._std_cost = 0
        self._other_costs = []
        self._cost = 0

        self._std_evo_constraints = []
        self._other_constraints = []
        self._constraints = []

        if self.model.mld_numeric is not None:
            self._sys_evo_matrices = MpcEvoMatrices(self)
            self._variables: MpcVariables = MpcVariables(mpc_controller=self, x_k=x_k, omega_tilde_k=omega_tilde_k)
            self._std_obj_weights = MpcObjectiveWeights(self)
        else:
            self._sys_evo_matrices = None
            self._variables: MpcVariables = None
            self._std_obj_weights = None

    def update_horizons(self, N_p=ParNotSet, N_tilde=ParNotSet, x_k=None, omega_k_tilde=None):
        old_N_p = self.N_p
        old_N_tilde = self.N_tilde

        self._N_p = N_p if N_p is not ParNotSet else self.N_p or 0
        self._N_tilde = N_tilde if N_tilde is not ParNotSet else N_p + 1

        if old_N_p!=self._N_p or old_N_tilde!=self._N_tilde:
            self._update_components()

    @property
    def variables(self) -> MpcVariables:
        return self._variables

    @property
    def variables_k(self):
        if self._variables:
            return self._variables.variables_k
        else:
            return MpcVariables.VarStruct_k()

    @property
    def variables_k_neg1(self):
        if self._variables:
            return self._variables.variables_k_neg1
        else:
            return MpcVariables.VarStruct_k_neg1()

    @variables_k_neg1.setter
    def variables_k_neg1(self, variables_k_neg1_struct):
        if self._variables:
            self._variables.variables_k_neg1=variables_k_neg1_struct
        else:
            raise AttributeError("MpcController 'variables' attribute has not been initialised.")

    @property
    def sys_evo_matrices(self):
        return self._sys_evo_matrices

    @property
    def std_obj_weights(self):
        return self._std_obj_weights

    @property
    def cost(self):
        return self._cost

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

    @build_required_decor
    def set_std_obj_weights(self, objective_weights_struct=None, **kwargs):
        if self._std_obj_weights:
            self._std_obj_weights.set(objective_weights_struct=objective_weights_struct, **kwargs)
        else:
            self._std_obj_weights = MpcObjectiveWeights(self, objective_weights_struct=objective_weights_struct,
                                                        **kwargs)

    @build_required_decor
    def update_std_obj_weights(self, objective_weights_struct=None, **kwargs):
        if self._std_obj_weights:
            self._std_obj_weights.update(objective_weights_struct, **kwargs)
        else:
            self._std_obj_weights = MpcObjectiveWeights(self, objective_weights_struct=objective_weights_struct,
                                                        **kwargs)

    def gen_std_type_cost(self, objective_weights: MpcObjectiveWeights, variables: MpcVariables):
        try:
            return objective_weights.gen_cost(variables)
        except AttributeError:
            if objective_weights is None:
                return 0
            else:
                raise TypeError(f"objective_weights must be of type {MpcObjectiveWeights.__class__.__name__}")

    @build_required_decor
    def set_costs(self, std_cost=ParNotSet, other_costs=ParNotSet):
        if std_cost is not ParNotSet:
            self._std_cost = std_cost if std_cost is not None else self.gen_std_type_cost(self.std_obj_weights,
                                                                                          self.variables)
        if other_costs is not ParNotSet:
            self._other_costs = other_costs if other_costs is not None else []

        self._cost = self._std_cost + cvx.sum(self._other_costs)

        self.set_build_required()

    def gen_evo_mats(self):
        pass

    def _gen_std_evo_constraints(self):
        std_evo_constraints = self.gen_evo_constraints(self.x_k, self.omega_tilde_k)
        if std_evo_constraints is not None:
            return [std_evo_constraints]
        else:
            return []

    def gen_evo_constraints(self, x_k=None, omega_tilde_k=None,
                            N_p=ParNotSet, N_tilde=ParNotSet, mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet,
                            sys_evo_matrices=ParNotSet):
        # H_v @ v_tilde <= H_x @ x_0 + H_omega @ omega_tilde + H_5

        x_k = x_k if x_k is not None else self.x_k
        omega_tilde_k = omega_tilde_k if omega_tilde_k is not None else self.omega_tilde_k

        if sys_evo_matrices is ParNotSet:
            mld_numeric_k = mld_numeric_k if mld_numeric_k is not self.mld_numeric_k else ParNotSet
            mld_numeric_tilde = mld_numeric_tilde if mld_numeric_tilde is not self.mld_numeric_tilde else ParNotSet
            N_p = N_p if N_p != self.N_p else ParNotSet
            N_tilde = N_tilde if N_tilde != self.N_tilde else ParNotSet

            if not eq_all_val(mld_numeric_k, mld_numeric_tilde, N_p, N_tilde, val=ParNotSet):
                sys_evo_matrices = MpcEvoMatrices(self, N_p=N_p, N_tilde=N_tilde,
                                                  mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)
            else:
                sys_evo_matrices = self._sys_evo_matrices

        if sys_evo_matrices is not None:
            LHS = (matmul(sys_evo_matrices.constraint['H_v_N_tilde'], self._variables['v']['var_N_tilde']))
            RHS = (matmul(sys_evo_matrices.constraint['H_x_N_tilde'], x_k) +
                   matmul(sys_evo_matrices.constraint['H_omega_N_tilde'], omega_tilde_k)
                   + sys_evo_matrices.constraint['H_5_N_tilde'])

            return LHS <= RHS
        else:
            return None

    def set_constraints(self, std_evo_constaints=ParNotSet, other_constraints=ParNotSet,
                        disable_soft_constraints=False):
        if std_evo_constaints is not ParNotSet:
            self._std_evo_constraints = (std_evo_constaints if std_evo_constaints is not None else
                                         self._gen_std_evo_constraints())
        if other_constraints is not ParNotSet:
            self._other_constraints = other_constraints if other_constraints is not None else []

        if disable_soft_constraints:
            soft_cons = [
                self.variables.mu.var_N_tilde == np.zeros(self.variables.mu.var_N_tilde.shape)
            ]
        else:
            soft_cons = []

        self._constraints = self._std_evo_constraints + self._other_constraints + soft_cons


    def set_build_required(self):
        self._build_required = True

    @property
    def build_required(self):
        return self._build_required or self.has_updated_version()

    @build_required_decor(set=False)
    def build(self, with_std_cost=True, with_std_constraints=True, sense=None, disable_soft_constraints=False):
        if with_std_cost:
            std_cost = None
        else:
            std_cost = 0

        self.set_costs(std_cost=std_cost)

        if with_std_constraints:
            std_evo_constraints = None
        else:
            std_evo_constraints = []

        self.set_constraints(std_evo_constaints=std_evo_constraints, disable_soft_constraints=disable_soft_constraints)
        assert isinstance(self._constraints, list)

        sense = 'minimize' if sense is None else sense
        if sense.lower().startswith('min'):
            self._problem = cvx.Problem(cvx.Minimize(self._cost), self._constraints)
        elif sense.lower().startswith('max'):
            self._problem = cvx.Problem(cvx.Maximize(self._cost), self._constraints)
        else:
            raise ValueError(f"Problem 'sense' must be either 'minimize' or 'maximize', got '{sense}'.")

        self.update_stored_version()

    def solve(self, solver=None,
              ignore_dcp=False,
              warm_start=True,
              verbose=False,
              parallel=False, *, method=None, **kwargs):

        if self.build_required:
            raise RuntimeError("Mpc problem has not been built or needs to be rebuilt.")

        try:
            solution = self._problem.solve(solver=solver,
                                           ignore_dcp=ignore_dcp,
                                           warm_start=warm_start,
                                           verbose=verbose,
                                           parallel=parallel, method=method, **kwargs)
        except cvx.error.SolverError as se:
            with io.StringIO() as std_out_redirect:
                if verbose == False:
                    with contextlib.redirect_stdout(std_out_redirect):
                        try:
                            self._problem.solve(solver=solver,
                                                ignore_dcp=ignore_dcp,
                                                warm_start=warm_start,
                                                verbose=True,
                                                parallel=parallel, method=method, **kwargs)
                        except:
                            pass
                        std_out_redirect.seek(0)
                        out = std_out_redirect.read()
                else:
                    out = ""

            raise RuntimeError(f"{se.args[0]}\n\n{out}") from se

        if not np.isfinite(solution):
            raise RuntimeError(f"solve() failed with objective: '{solution}', and status: {self._problem.status}")
        else:
            return solution

    def feedback(self, x_k=None, omega_tilde_k=None,
                 solver=None,
                 ignore_dcp=False, warm_start=True, verbose=False,
                 parallel=False, *, method=None, **kwargs
                 ):
        if x_k is not None:
            self.x_k = x_k
        if omega_tilde_k is not None:
            self.omega_tilde_k = omega_tilde_k

        self.solve(solver=solver,
                   ignore_dcp=ignore_dcp, warm_start=warm_start, verbose=verbose,
                   parallel=parallel, method=method, **kwargs)

        return self.variables_k

    def __repr__(self):
        repr_dict = {data_type: getattr(self, data_type, ParNotSet) for data_type in self._data_types}
        return struct_repr(repr_dict, type_name=self.__class__.__name__, align_values=True, align_padding_width=1,
                           value_format_str='\b{value}')

    # def __dir__(self):
    #     super_dir = super(MpcController, self).__dir__()
    #     new_dir = set([item for item in super_dir if item[0] != '_' or item[1] == '_'])
    #     return sorted(new_dir)
