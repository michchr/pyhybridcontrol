import cvxpy as cvx
import numpy as np

import cvxpy.expressions.expression as cvx_e

from controllers.mpc_controller.mpc_components import MpcEvoMatrices, MpcOptimizationVars, MpcObjectiveWeights
from controllers.mpc_controller.mpc_utils import process_base_args
from models.mld_model import MldModel, MldInfo, MldSystemModel
from utils.decorator_utils import process_method_args_decor
from utils.matrix_utils import atleast_2d_col, matmul
from structdict import struct_prop_fixed_dict, struct_repr
from utils.helper_funcs import is_all_val

from utils.func_utils import ParNotSet


class MpcBase:
    def __init__(self, agent=None, N_p=None, N_tilde=None, model=None, mld_numeric_k=None):
        if agent is not None:
            self._agent = agent
            self._N_p = None
            self._N_tilde = None
            self._mld_numeric_tilde = None
            self._model = self._agent._agent_model
        else:
            self._model = model if model is not None else MldSystemModel(mld_numeric=mld_numeric_k)
            self._agent = None
            self._N_p = N_p or 0
            self._mld_numeric_tilde = None

    @property
    def N_p(self):
        return (self._agent.N_p if self._agent else self._N_p)

    @property
    def N_tilde(self):
        return (self._agent.N_tilde if self._agent else self._N_tilde)

    @property
    def mld_numeric_k(self) -> MldModel:
        mld_numeric_tilde = self._mld_numeric_tilde
        return mld_numeric_tilde[0] if mld_numeric_tilde else self._model.mld_numeric

    @property
    def mld_numeric_tilde(self):
        return self._mld_numeric_tilde

    @property
    def mld_info_k(self) -> MldInfo:
        return self.mld_numeric_k.mld_info


class MpcController(MpcBase):
    _data_types = ['_opt_vars', '_std_obj_weights']

    FeedBackStruct = struct_prop_fixed_dict('FeedBackStruct',
                                            [var_name + "_k" for var_name in MpcOptimizationVars._var_names])

    def __init__(self, agent=None, N_p=None, N_tilde=None,
                 model: MldSystemModel = None, mld_numeric: MldModel = None,
                 x_k=None, omega_tilde_k=None):
        super(MpcController, self).__init__(agent=agent, N_p=N_p, N_tilde=N_tilde,
                                            mld_numeric_k=mld_numeric, model=model)

        self._x_k = None
        self._omega_tilde_k = None
        self.set_x_k(x_k=x_k)
        self.set_omega_tilde_k(omega_tilde_k=omega_tilde_k)

        self._sys_evo_matrices = MpcEvoMatrices(self)
        self._opt_vars = MpcOptimizationVars(self)

        self._std_obj_weights = MpcObjectiveWeights(self)
        self._std_cost = None
        self._custom_cost = None
        self._cost = None

        self._std_evo_constraints = []
        self._custom_constraints = []
        self._constraints = []
        self._build_required = True

    @property
    def opt_vars(self):
        return self._opt_vars

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
    def problem(self)->cvx.Problem:
        return self._problem

    @property
    def x_k(self):
        return self._x_k

    @x_k.setter
    def x_k(self, value):
        self.set_x_k(x_k=value)

    @property
    def omega_tilde_k(self) -> cvx.Parameter:
        return self._omega_tilde_k

    @omega_tilde_k.setter
    def omega_tilde_k(self, value):
        self.set_omega_tilde_k(omega_tilde_k=value)

    @process_method_args_decor(process_base_args)
    def set_x_k(self, x_k=None, mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None):

        required_shape = (mld_info_k.nx, 1)
        updated_param = self._process_parameter(name="x_k", parameter=self._x_k,
                                                required_shape=required_shape,
                                                new_value=x_k)
        if updated_param is not None:
            self._x_k = updated_param
            self.set_build_required()

    @process_method_args_decor(process_base_args)
    def set_omega_tilde_k(self, omega_tilde_k=None, N_tilde=None,
                          mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None):
        required_shape = (mld_info_k.nomega * N_tilde, 1)
        updated_param = self._process_parameter(name="omega_tilde_k", parameter=self._omega_tilde_k,
                                                required_shape=required_shape,
                                                new_value=omega_tilde_k)
        if updated_param is not None:
            self._omega_tilde_k = updated_param
            self.set_build_required()

    def _process_parameter(self, name, parameter, required_shape, new_value=None):
        if 0 not in required_shape:
            if new_value is not None:
                if isinstance(new_value, cvx_e.Expression):
                    set_value = new_value
                    if set_value.shape == required_shape:
                        return set_value
                else:
                    set_value = atleast_2d_col(new_value)
                    if set_value.dtype == np.object_:
                        raise TypeError(
                            f"'new_value' must be a subclass of a cvxpy {cvx_e.Expression.__name__}, an numeric array "
                            f"like object or None.")

                if set_value.shape != required_shape:
                    raise ValueError(
                        f"Incorrect shape:{set_value.shape} for {name}, a shape of {required_shape} is required.")
            else:
                set_value = None

            if parameter is None or parameter.shape != required_shape:
                if set_value is None:
                    if parameter is not None and not isinstance(parameter, cvx.Parameter):
                        raise ValueError(
                            f"'{name}' is currently a '{parameter.__class__.__name__}' object and can therefore not be "
                            f"automatically set to a zero 'cvx.Parameter' object. It needs to be set explicitly.")
                    set_value = np.zeros((required_shape))
                return cvx.Parameter(shape=required_shape, name=name, value=set_value)
            elif set_value is not None:
                if isinstance(parameter, cvx.Parameter):
                    parameter.value = set_value
                else:
                    return cvx.Parameter(shape=required_shape, name=name, value=set_value)
        else:
            return np.empty(required_shape)


    def set_std_obj_weights(self, objective_weights_struct=None, **kwargs):
        self._std_obj_weights.set(objective_weights_struct=objective_weights_struct, **kwargs)
        self.set_build_required()

    def update_std_obj_weights(self, objective_weights_struct=None, **kwargs):
        self._std_obj_weights.update(objective_weights_struct, **kwargs)
        self.set_build_required()

    def gen_std_type_cost(self, objective_weights: MpcObjectiveWeights, opt_vars: MpcOptimizationVars):
        try:
            return objective_weights.gen_cost(opt_vars)
        except AttributeError:
            raise TypeError(f"objective_weights must be of type {MpcObjectiveWeights.__class__.__name__}")


    def gen_evo_mats(self):
        pass


    def _gen_std_evo_constraints(self):
        return [self.gen_evo_constraints(self.x_k, self.omega_tilde_k)]

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

            if not is_all_val(mld_numeric_k, mld_numeric_tilde, N_p, N_tilde, val=ParNotSet):
                sys_evo_matrices = MpcEvoMatrices(self, N_p=N_p, N_tilde=N_tilde,
                                                  mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)
            else:
                sys_evo_matrices = self._sys_evo_matrices

        LHS = (matmul(sys_evo_matrices.constraint['H_v_N_tilde'], self._opt_vars['v']['var_N_tilde']))
        RHS = (matmul(sys_evo_matrices.constraint['H_x_N_tilde'], x_k) +
               matmul(sys_evo_matrices.constraint['H_omega_N_tilde'], omega_tilde_k)
               + sys_evo_matrices.constraint['H_5_N_tilde'])

        return LHS <= RHS

    def set_build_required(self):
        self._build_required = True

    def build(self, with_std_cost=True, with_std_constraints=True, sense=None):
        if with_std_cost:
            self._std_cost = self.gen_std_type_cost(self._std_obj_weights, self._opt_vars)
        else:
            self._std_cost = 0

        custom_cost = self._custom_cost if self._custom_cost is not None else 0
        self._cost = self._std_cost + custom_cost

        if with_std_constraints:
            self._std_evo_constraints = self._gen_std_evo_constraints()
        else:
            self._std_evo_constraints = []

        self._constraints = self._std_evo_constraints + self._custom_constraints
        assert isinstance(self._constraints, list)

        sense = 'minimize' if sense is None else sense
        if sense.lower().startswith('min'):
            self._problem = cvx.Problem(cvx.Minimize(self._cost), self._constraints)
        elif sense.lower().startswith('max'):
            self._problem = cvx.Problem(cvx.Maximize(self._cost), self._constraints)
        else:
            raise ValueError(f"Problem 'sense' must be either 'minimize' or 'maximize', got '{sense}'.")

        self._build_required = False

    def solve(self, solver=None,
              ignore_dcp=False,
              warm_start=True,
              verbose=False,
              parallel=False, *, method=None, **kwargs):

        if self._build_required:
            raise RuntimeError("Mpc problem has not been built or needs to be rebuilt.")

        solution = self._problem.solve(solver=solver,
                                       ignore_dcp=ignore_dcp,
                                       warm_start=warm_start,
                                       verbose=verbose,
                                       parallel=parallel, method=method, **kwargs)

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

        if self._build_required:
            self.build()

        self.solve(solver=solver,
                   ignore_dcp=ignore_dcp, warm_start=warm_start, verbose=verbose,
                   parallel=parallel, method=method, **kwargs)

        feedback_struct = self.FeedBackStruct()
        for var_name_k in feedback_struct.keys():
            var_name = var_name_k.rstrip('_k')
            opt_var = self._opt_vars[var_name]['var_mat_N_tilde']
            if isinstance(opt_var, cvx_e.Expression):
                feedback_struct[var_name_k] = self._opt_vars[var_name]['var_mat_N_tilde'].value[:, :1]
            else:
                feedback_struct[var_name_k] = self._opt_vars[var_name]['var_mat_N_tilde'][:, :1]

        return feedback_struct

    def __repr__(self):
        def value_repr(value): return (
            struct_repr(value, type_name='', repr_format_str='{type_name}{{{key_arg}{items}}}', align_values=True))

        repr_dict = {data_type: value_repr(getattr(self, data_type, None)) for data_type in
                     self._data_types}
        return struct_repr(repr_dict, type_name=self.__class__.__name__, align_values=True, align_padding_width=1,
                           value_format_str='\b{value}')
