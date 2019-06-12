import cvxpy as cvx
import numpy as np
from cvxpy.expressions import expression as cvx_e

from controllers.components.component_base import ComponentBase
from controllers.controller_utils import process_base_args
from models.mld_model import MldInfo, MldModel
from structdict import named_fixed_struct_dict, StructDict
from utils.decorator_utils import process_method_args_decor
from utils.func_utils import ParNotSet
from utils.helper_funcs import is_all_None
from utils.matrix_utils import matmul, atleast_2d_col

EvoVariableStruct = named_fixed_struct_dict('EvoVariableStruct',
                                            ['var_k', 'var_N_tilde', 'var_N_p', 'var_mat_N_tilde', 'var_k_neg1',
                                            'var_dim'],
                                            sorted_repr=False)

_var_names = MldInfo._var_names
VariablesStruct_k = named_fixed_struct_dict('VariablesStruct_k', _var_names)
VariablesStruct_k_neg1 = named_fixed_struct_dict('VariablesStruct_k_neg1', _var_names)


class EvoVariables(ComponentBase):
    _controllable_vars = MldInfo._controllable_var_names
    _concat_controllable_vars = MldInfo._concat_controllable_var_names
    _optimization_vars = _controllable_vars + _concat_controllable_vars

    _state_output_vars = MldInfo._state_var_names + MldInfo._output_var_names

    _var_names = _var_names
    _field_names = _var_names
    _field_names_set = frozenset(_field_names)

    VariablesStruct_k = VariablesStruct_k
    VariablesStruct_k_neg1 = VariablesStruct_k_neg1

    __internal_names = ['_x_k', '_custom_variables']
    _internal_names_set = set(__internal_names)

    def __init__(self, controller=None, x_k=None, omega_tilde_k=None,
                 N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet):
        super(EvoVariables, self).__init__(controller, N_p=N_p, N_tilde=N_tilde,
                                           mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)

        self.update(x_k=x_k, omega_tilde_k=omega_tilde_k)

    def _reset(self):
        super(EvoVariables, self)._reset()
        self._base_dict_init({var_name: EvoVariableStruct() for var_name in self._var_names})
        self._custom_variables = []
        self._x_k = None

    @property
    def mld_evo_matrices(self):
        return self._controller.mld_evo_matrices

    @property
    def omega_tilde_k(self):
        return self.omega.var_N_tilde

    @omega_tilde_k.setter
    def omega_tilde_k(self, value):
        self.update(omega_tilde_k=value)

    @property
    def x_k(self):
        return self._x_k

    @x_k.setter
    def x_k(self, value):
        self.update(x_k=value)

    @property
    def variables_k(self):
        vars_k = self.VariablesStruct_k()
        for var_name, var in self.items():
            var_k = var.var_k
            if var_k is not None:
                if isinstance(var_k, cvx_e.Expression):
                    vars_k[var_name] = var_k.value
                else:
                    vars_k[var_name] = var_k
        return vars_k

    @property
    def variables_k_neg1(self):
        vars_k_neg1 = self.VariablesStruct_k_neg1()
        for var_name, var in self.items():
            var_k_neg1 = var.var_k_neg1
            if var_k_neg1 is not None:
                if isinstance(var_k_neg1, cvx_e.Expression):
                    vars_k_neg1[var_name] = var_k_neg1.value
                else:
                    vars_k_neg1[var_name] = var_k_neg1
        return vars_k_neg1

    @variables_k_neg1.setter
    def variables_k_neg1(self, variables_k_neg1_struct):
        variables_k_neg1_struct = variables_k_neg1_struct or self.VariablesStruct_k_neg1()
        self.update(variables_k_neg1_struct=variables_k_neg1_struct)


    #todo still needs work
    def add_custom_var(self, var_name, var_N_tilde, N_p=None, N_tilde=None):
        if var_name in self:
            raise ValueError(f"Variable with name:{var_name} already exists.")
        elif not isinstance(var_N_tilde, cvx_e.Expression):
            raise TypeError(f"Variable must be a subtype of class {cvx_e.Expression.__name__}.")
        else:
            N_tilde = N_tilde if N_tilde is not None else self.N_tilde
            N_p = N_p if N_p is not None else self.N_p
            if var_N_tilde.size % N_tilde:
                raise ValueError(f"var_N_tilde.size: {var_N_tilde.size} must be a multiple of N_tilde: {N_tilde}")
            var_dim = var_N_tilde.size // N_tilde
            self[var_name] = EvoVariableStruct()
            self._set_var_using_var_N_tilde(variable=self[var_name],
                                            var_dim=var_dim, var_N_tilde=var_N_tilde, N_tilde=N_tilde, N_p=N_p)

            self.update()

    # increments version conditionally hence no version increment decorator
    @process_method_args_decor(process_base_args)
    def update(self, x_k=None, omega_tilde_k=None, N_p=None, N_tilde=None,
               mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
               mld_info_k: MldInfo = None, variables_k_neg1_struct=None):

        x_k_update = self._set_x_k(_disable_process_args=True,
                                   x_k=x_k,
                                   N_p=N_p, N_tilde=N_tilde,
                                   mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                   mld_info_k=mld_info_k)

        if x_k_update is not None:
            self._x_k = x_k_update

        omega_var_update = self._set_omega_var(_disable_process_args=True,
                                               omega_tilde_k=omega_tilde_k,
                                               N_p=N_p, N_tilde=N_tilde,
                                               mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                               mld_info_k=mld_info_k)

        if omega_var_update is not None:
            self._base_dict_update(omega=omega_var_update)

        if self.has_updated_version(sub_object_names=('mld_numeric_k', 'mld_numeric_tilde')):
            variables_update = (
                self.gen_optimization_vars(_disable_process_args=True,
                                           N_p=N_p, N_tilde=N_tilde,
                                           mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                           mld_info_k=mld_info_k))
            self._base_dict_update(variables_update)
            self.update_stored_version()
        else:
            variables_update = None

        if not is_all_None(x_k_update, omega_var_update, variables_update):
            state_output_vars_update = (
                self.gen_state_output_vars(_disable_process_args=True,
                                           variables=self,
                                           x_k=self.x_k, omega_tilde_k=self.omega_tilde_k,
                                           N_p=N_p, N_tilde=N_tilde,
                                           mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                           mld_info_k=mld_info_k))

            self._base_dict_update(state_output_vars_update)
        else:
            state_output_vars_update = None

        variables_k_neg1_struct_update = (
            self._set_vars_k_neg1(_disable_process_args=True,
                                  variables_k_neg1_struct=variables_k_neg1_struct,
                                  N_p=N_p, N_tilde=N_tilde,
                                  mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                  mld_info_k=mld_info_k))

        if variables_k_neg1_struct_update is not None:
            for var_name, var_k_neg1 in variables_k_neg1_struct_update.items():
                self[var_name].var_k_neg1 = var_k_neg1

        if not is_all_None(x_k_update, omega_var_update, variables_update, state_output_vars_update,
                           variables_k_neg1_struct_update):
            self.increment_version()
            self.set_build_required()

        self._update_set_with(N_p, N_tilde)

    @process_method_args_decor(process_base_args)
    def gen_optimization_vars(self, N_p=None, N_tilde=None,
                              mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
                              mld_info_k: MldInfo = None):

        slack_names = mld_info_k._slack_var_names

        # extract variable matrix_types from mld mld_infos
        if mld_numeric_tilde:
            var_types_N_tilde = {
                var_name: (
                    np.vstack(
                        [mld_numeric_tilde[k].mld_info.get_var_type(var_name) for k in range(N_tilde)]
                    ) if mld_info_k.get_var_dim(var_name) else (
                        np.empty((0, mld_info_k.get_var_dim(var_name) * N_tilde), dtype=np.str))
                ) for var_name in self._controllable_vars
            }
        else:
            var_types_N_tilde = {
                var_name: (
                    np.tile(mld_info_k.get_var_type(var_name), (N_tilde, 1)))
                for var_name in self._controllable_vars
            }

        def to_bin_index(type_mat):
            return (list(map(tuple, np.argwhere(type_mat == 'b').tolist())))

        # generate individual variable tilde mats
        opt_var_N_tilde = {
            var_name: (
                cvx.Variable(var_type_mat.shape, boolean=to_bin_index(var_type_mat),
                             name="".join([var_name.capitalize(), '_var_N_tilde']),
                             nonneg=(var_name in slack_names or None)) if var_type_mat.size
                else np.empty((0, 1))
            ) for var_name, var_type_mat in var_types_N_tilde.items()
        }

        variables = StructDict({var_name: EvoVariableStruct() for var_name in self._optimization_vars})
        for var_name in self._controllable_vars:
            self._set_var_using_var_N_tilde(variable=variables[var_name],
                                            var_dim=mld_info_k.get_var_dim(var_name),
                                            var_N_tilde=opt_var_N_tilde[var_name],
                                            N_p=N_p, N_tilde=N_tilde)

        # add combined input variable tilde mat
        opt_var_mats_N_tilde = [variables[var_name].var_mat_N_tilde for var_name in self._controllable_vars if
                                mld_info_k.get_var_dim(var_name)]

        v_var_mat_N_tilde = cvx.vstack(opt_var_mats_N_tilde) if opt_var_mats_N_tilde else np.empty((0, N_tilde))
        self._set_var_using_var_mat_N_tilde(variable=variables['v'],
                                            var_dim=mld_info_k.nv,
                                            var_mat_N_tilde=v_var_mat_N_tilde,
                                            N_p=N_p, N_tilde=N_tilde)

        return variables

    @process_method_args_decor(process_base_args)
    def gen_state_output_vars(self, variables=None,
                              x_k=None, omega_tilde_k=None,
                              N_p=None, N_tilde=None,
                              mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
                              mld_info_k: MldInfo = None):

        state_output_vars = StructDict({var_name: EvoVariableStruct() for var_name in self._state_output_vars})

        x_k = x_k if x_k is not None else self.x_k
        omega_tilde_k = omega_tilde_k if omega_tilde_k is not None else self.omega_tilde_k

        variables = variables or self

        if mld_info_k.nx:
            state_output_vars.x.var_N_tilde = (
                    matmul(self.mld_evo_matrices.state_input['Phi_x_N_tilde'], x_k) +
                    matmul(self.mld_evo_matrices.state_input['Gamma_v_N_tilde'], variables.v.var_N_tilde) +
                    matmul(self.mld_evo_matrices.state_input['Gamma_omega_N_tilde'], omega_tilde_k) +
                    self.mld_evo_matrices.state_input['Gamma_5_N_tilde']
            )
        else:
            state_output_vars.x.var_N_tilde = np.empty((0, 1))

        if mld_info_k.ny:
            state_output_vars.y.var_N_tilde = (
                    matmul(self.mld_evo_matrices.output['L_x_N_tilde'], x_k) +
                    matmul(self.mld_evo_matrices.output['L_v_N_tilde'], variables['v']['var_N_tilde']) +
                    matmul(self.mld_evo_matrices.output['L_omega_N_tilde'], omega_tilde_k) +
                    self.mld_evo_matrices.output['L_5_N_tilde']
            )
        else:
            state_output_vars.y.var_N_tilde = np.empty((0, 1))

        for var_name in self._state_output_vars:
            var_N_tilde = state_output_vars[var_name].var_N_tilde
            self._set_var_using_var_N_tilde(variable=state_output_vars[var_name],
                                            var_dim=mld_info_k.get_var_dim(var_name),
                                            var_N_tilde=var_N_tilde,
                                            N_p=N_p, N_tilde=N_tilde)

        return state_output_vars

    @staticmethod
    def _set_var_using_var_N_tilde(variable, var_dim, var_N_tilde, N_p, N_tilde):
        variable.var_N_tilde = var_N_tilde
        variable.var_k = var_N_tilde[:var_dim, :]

        if N_p <= N_tilde:
            variable.var_N_p = var_N_tilde[:var_dim * N_p, :]
        else:
            variable.var_N_p = None

        variable.var_mat_N_tilde = (
            cvx.reshape(var_N_tilde, (var_dim, N_tilde)) if var_N_tilde.size else np.empty((0, N_tilde)))
        variable.var_dim = var_dim
        return variable

    @staticmethod
    def _set_var_using_var_mat_N_tilde(variable, var_dim, var_mat_N_tilde, N_p, N_tilde):
        variable.var_mat_N_tilde = var_mat_N_tilde
        variable.var_N_tilde = var_N_tilde = (
            cvx.reshape(var_mat_N_tilde, (var_dim * N_tilde, 1)) if var_mat_N_tilde.size else np.empty((0, 1)))

        variable.var_k = var_N_tilde[:var_dim, :]

        if N_p <= N_tilde:
            variable.var_N_p = var_N_tilde[:var_dim * N_p, :]
        else:
            variable.var_N_p = None

        variable.var_dim = var_dim
        return variable

    def get_variables_with(self, x_k=None, omega_tilde_k=None):
        variables = self.copy()
        state_output_vars = self.gen_state_output_vars(variables, x_k=x_k, omega_tilde_k=omega_tilde_k)
        variables._base_dict_update(state_output_vars)
        return variables

    @process_method_args_decor(process_base_args)
    def _set_x_k(self, x_k, N_p=None, N_tilde=None,
                 mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
                 mld_info_k: MldInfo = None):
        required_shape = (mld_info_k.nx, 1)
        x_k = EvoVariables._process_parameter_update(name="x_k", parameter=self._x_k,
                                                     required_shape=required_shape,
                                                     new_value=x_k)
        return x_k

    @process_method_args_decor(process_base_args)
    def _set_omega_var(self, omega_tilde_k, N_p=None, N_tilde=None,
                       mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
                       mld_info_k: MldInfo = None):
        required_shape = (mld_info_k.nomega * N_tilde, 1)
        omega_tilde_k = self._process_parameter_update(name="omega_tilde_k", parameter=self.omega.var_N_tilde,
                                                       required_shape=required_shape,
                                                       new_value=omega_tilde_k)

        if omega_tilde_k is not None:
            omega_var = EvoVariableStruct()
            return self._set_var_using_var_N_tilde(variable=omega_var,
                                                   var_dim=mld_info_k.nomega,
                                                   var_N_tilde=omega_tilde_k,
                                                   N_p=N_p, N_tilde=N_tilde)
        else:
            return omega_tilde_k

    @process_method_args_decor(process_base_args)
    def _set_vars_k_neg1(self, variables_k_neg1_struct=None, N_p=None, N_tilde=None,
                         mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
                         mld_info_k: MldInfo = None):

        variables_k_neg1_struct = variables_k_neg1_struct or self.VariablesStruct_k_neg1.fromkeys(self)
        variables_k_neg1_struct_update = StructDict()
        for var_name, variable in self.items():
            var_k_neg1_update = variables_k_neg1_struct.get(var_name,None)
            var_update = (
                self._process_parameter_update(name=var_name + "_k_neg1", parameter=variable.var_k_neg1,
                                               required_shape=(variable.var_dim, 1),
                                               new_value=var_k_neg1_update))
            if var_update is not None:
                variables_k_neg1_struct_update[var_name] = var_update

        return variables_k_neg1_struct_update if variables_k_neg1_struct_update else None

    @staticmethod
    def _process_parameter_update(name, parameter, required_shape, new_value=None):
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
                return None
        elif parameter is None or parameter.shape != required_shape:
            return np.empty(required_shape)
        else:
            return None
