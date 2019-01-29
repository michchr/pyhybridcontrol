import itertools
import functools

from collections import namedtuple
import cvxpy as cvx
import numpy as np

from models.mld_model import MldModel, MldInfo, MldSystemModel
from utils.decorator_utils import ParNotSet, process_method_args_decor
from utils.matrix_utils import get_mat_ops, block_toeplitz, block_diag_dense, atleast_2d_col
from utils.structdict import StructDict, StructPropFixedDictMixin, struct_prop_fixed_dict, struct_repr


def _process_base_args(self, f_kwargs=None, *,
                       N_p=ParNotSet, N_tilde=ParNotSet,
                       mld_numeric=ParNotSet, mld_numeric_tilde=ParNotSet, mld_info_k0=ParNotSet):
    if N_p is None:
        f_kwargs['N_p'] = N_p = self.N_p

    if N_tilde is None:
        f_kwargs['N_tilde'] = N_p + 1 if N_p is not ParNotSet else self.N_p + 1

    if mld_numeric_tilde is None:
        f_kwargs['mld_numeric_tilde'] = mld_numeric_tilde = self.mld_numeric_tilde

    if mld_numeric is None:
        f_kwargs['mld_numeric'] = mld_numeric = self.mld_numeric

    if mld_info_k0 is None:
        f_kwargs['mld_info_k0'] = mld_numeric_tilde[0].mldinfo if mld_numeric_tilde else mld_numeric.mld_info

    return f_kwargs


def _process_mat_op_args(self, f_kwargs=None, *,
                         sparse=ParNotSet, mat_ops=ParNotSet):
    if sparse is None:
        f_kwargs['sparse'] = False

    if mat_ops is None:
        f_kwargs['mat_ops'] = get_mat_ops(sparse=f_kwargs['sparse'])

    return f_kwargs


def _process_A_pow_tilde_arg(self, f_kwargs=None, *,
                             A_pow_tilde=ParNotSet):
    if A_pow_tilde is None:
        f_kwargs['A_pow_tilde'] = (
            self._gen_A_pow_tilde(_disable_process_args=True, N_tilde=f_kwargs['N_tilde'],
                                  mld_numeric=f_kwargs['mld_numeric'], mld_numeric_tilde=f_kwargs['mld_numeric_tilde'],
                                  sparse=f_kwargs['sparse'], mat_ops=f_kwargs['mat_ops']))

    return f_kwargs


class MpcBase:
    def __init__(self, agent=None, N_p=None, N_tilde=None, model=None, mld_numeric=None):
        if agent is not None:
            self._agent = agent
            self._N_p = None
            self._N_tilde = None
            self._mld_numeric = None
            self._mld_numeric_tilde = None
            self._model = self._agent._agent_model
        else:
            self._model = model if model is not None else MldSystemModel(mld_numeric=mld_numeric)
            self._agent = None
            self._N_p = N_p or 0
            self._mld_numeric = None
            self._mld_numeric_tilde = None

    @property
    def N_p(self):
        return (self._agent.N_p if self._agent else self._N_p)

    @property
    def N_tilde(self):
        return (self._agent.N_tilde if self._agent else self._N_tilde)

    @property
    def mld_numeric(self) -> MldModel:
        return self._model._mld_numeric

    @property
    def mld_numeric_tilde(self):
        return None

    @property
    def mld_info_k0(self) -> MldInfo:
        return self.mld_numeric_tilde[0].mldinfo if self.mld_numeric_tilde else self.mld_numeric.mld_info


class MpcController(MpcBase):
    _data_types = ['_opt_vars', '_objective_weights']

    def __init__(self, agent=None, N_p=None, N_tilde=None,
                 model: MldSystemModel = None, mld_numeric: MldModel = None,
                 x0=None, omega_tilde_k0=None):
        super(MpcController, self).__init__(agent=agent, N_p=N_p, N_tilde=N_tilde,
                                            mld_numeric=mld_numeric, model=model)

        x_val = np.ones((self.mld_info_k0['nx'], 1)) * 39
        w_val = np.ones((self.mld_info_k0['nomega'] * self.N_tilde, 1))

        self._x0 = x0 or cvx.Parameter(x_val.shape, value=x_val) if x_val.size else x_val
        self._omega_tilde_k = omega_tilde_k0 or cvx.Parameter(w_val.shape, value=w_val)

        self._mpc_evo_mats = MpcEvolutionMatrices(self)
        self._opt_vars = MpcOptimizationVars(self)
        self._objective_weights = MpcObjectiveWeights(self)

        self._cost = None
        self._constraints = None

    def __repr__(self):
        def value_repr(value): return (
            struct_repr(value, type_name='', repr_format_str='{type_name}{{{key_arg}{items}}}', align_values=True))

        repr_dict = {data_type: value_repr(self.__getattribute__(data_type)) for data_type in
                     self._data_types}
        return struct_repr(repr_dict, type_name=self.__class__.__name__, align_values=True, align_padding_width=1,
                           value_format_str='\b{value}')

    @process_method_args_decor(_process_base_args)
    def set_omega_tilde(self, x0=None, omega_tilde_k=None, N_p=None, N_tilde=None, mld_numeric=None,
                        mld_numeric_tilde=None,
                        mld_info_k0=None):
        pass

    @process_method_args_decor(_process_base_args)
    def gen_optimization_vars(self, N_p=None, N_tilde=None, mld_numeric=None, mld_numeric_tilde=None,
                              mld_info_k0=None):
        pass

    def set_constraints(self, x0=None, omega_tilde_k=None, N_p=None, N_tilde=None, mld_numeric=None,
                        mld_numeric_tilde=None, mld_info_k0=None):
        # H_v @ v_tilde <= H_x @ x_0 + H_omega @ omega_tilde + H_5

        LHS = (self._mpc_evo_mats.constraint['H_v_N_tilde'] @ self._opt_vars['v']['var_N_tilde'])
        RHS = (self._mpc_evo_mats.constraint['H_x_N_tilde'] * self._x0 +
               self._mpc_evo_mats.constraint['H_omega_N_tilde'] @ self._omega_tilde_k + self._mpc_evo_mats.constraint[
                   'H_5_N_tilde'])

        self._constraints = [LHS <= RHS]

    def set_cost(self, x0=None, omega_tilde_k=None):
        self._objective_weights.set(q_u=np.random.random((self.mld_info_k0['nu'], 1)))
        self._cost = self._objective_weights.get_cost()

    def gen_problem(self):
        pass

    def solve(self):
        pass


class MpcComponentsBase(StructPropFixedDictMixin, dict):
    _field_names = ()
    _field_names_set = frozenset()

    __internal_names = ['_mpc_controller', 'N_p', 'N_tilde', 'mld_numeric', 'mld_numeric_tilde', 'set_with_N_p',
                        'set_with_N_tilde']
    _internal_names_set = set(__internal_names)

    def __init__(self, mpc_controller: MpcController):
        self._mpc_controller: MpcController = mpc_controller
        self._reset()

    def _reset(self):
        self.set_with_N_p = None
        self.set_with_N_tilde = None

    @property
    def N_p(self):
        return self._mpc_controller.N_p

    @property
    def N_tilde(self):
        return self._mpc_controller.N_tilde

    @property
    def mld_numeric(self) -> MldModel:
        return self._mpc_controller.mld_numeric

    @property
    def mld_numeric_tilde(self):
        return self._mpc_controller.mld_numeric_tilde

    @property
    def mld_info_k0(self) -> MldInfo:
        return self._mpc_controller.mld_info_k0

    @property
    def x0(self):
        return self._mpc_controller._x0

    @property
    def omega_tilde_k(self):
        return self._mpc_controller._omega_tilde_k


class MpcEvolutionMatrices(MpcComponentsBase):
    _mpc_evo_mat_types_names = ['state_input', 'output', 'constraint']
    _MldModelMatTypesNamedTup = namedtuple('matrix_types', _mpc_evo_mat_types_names)
    matrix_types = _MldModelMatTypesNamedTup._make(_mpc_evo_mat_types_names)

    _var_lengths = ['N_p', 'N_tilde']
    _state_input_evo_mats = ['Phi_x', 'Gamma_v', 'Gamma_omega', 'Gamma_5']
    _output_evo_mats = ['L_x', 'L_v', 'L_omega', 'L_5']
    _constraint_evo_mats = ['H_x', 'H_v', 'H_omega', 'H_5']

    _state_evolution_mat_names = [name + '_' + lenght for name, lenght in
                                  itertools.product(_state_input_evo_mats, _var_lengths)]
    _output_evolution_mat_names = [name + '_' + lenght for name, lenght in
                                   itertools.product(_output_evo_mats, _var_lengths)]
    _constraint_evolution_mat_names = [name + '_' + lenght for name, lenght in
                                       itertools.product(_constraint_evo_mats, _var_lengths)]

    StateInputEvoMatStruct = struct_prop_fixed_dict('StateInputEvoMatStruct', _state_evolution_mat_names)
    OutputEvoMatStruct = struct_prop_fixed_dict('StateInputEvoMatStruct', _output_evolution_mat_names)
    ConstraintEvoMatStruct = struct_prop_fixed_dict('ConstraintEvoMatStruct',
                                                    _constraint_evolution_mat_names)

    _field_names = matrix_types
    _field_names_set = frozenset(_field_names)

    def _reset(self):
        super(MpcEvolutionMatrices, self)._reset()
        self._base_dict_init({
            self.matrix_types.state_input: self.StateInputEvoMatStruct(),
            self.matrix_types.output     : self.OutputEvoMatStruct(),
            self.matrix_types.constraint : self.ConstraintEvoMatStruct()
        })

    def __init__(self, mpc_controller: MpcController):
        super(MpcEvolutionMatrices, self).__init__(mpc_controller)
        self.update()

    @property
    def state_input(self):
        return self[self.matrix_types.state_input]

    @property
    def output(self):
        return self[self.matrix_types.output]

    @property
    def constraint(self):
        return self[self.matrix_types.constraint]

    @process_method_args_decor(_process_base_args, _process_mat_op_args, _process_A_pow_tilde_arg)
    def update(self, N_p=None, N_tilde=None,
               mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
               A_pow_tilde=None, sparse=None, mat_ops=None):

        self.gen_mpc_evolution_matrices(_disable_process_args=True, N_p=N_p, N_tilde=N_tilde,
                                        mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                        mld_info_k0=mld_info_k0,
                                        A_pow_tilde=A_pow_tilde,
                                        sparse=sparse, mat_ops=mat_ops)

        self.set_with_N_p = N_p
        self.set_with_N_tilde = N_tilde

    @process_method_args_decor(_process_base_args, _process_mat_op_args, _process_A_pow_tilde_arg)
    def gen_mpc_evolution_matrices(self, N_p=None, N_tilde=None,
                                   mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                                   A_pow_tilde=None, sparse=None, mat_ops=None):

        gen_kwargs = dict(_disable_process_args=True, N_tilde=N_tilde,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde, mld_info_k0=mld_info_k0,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        self[self.matrix_types.state_input].update(self.gen_state_input_evolution_matrices(N_p=N_p, **gen_kwargs))
        self[self.matrix_types.output].update(
            self.gen_output_evolution_matrices(N_p=N_p,
                                               state_input_evolution_struct=self[self.matrix_types.state_input],
                                               **gen_kwargs))

        self[self.matrix_types.constraint].update(
            self.gen_cons_evolution_matrices(N_p=N_p,
                                             state_input_evolution_struct=self[self.matrix_types.state_input],
                                             output_evolution_struct=self[self.matrix_types.output], **gen_kwargs))

    @process_method_args_decor(_process_base_args, _process_mat_op_args, _process_A_pow_tilde_arg)
    def gen_state_input_evolution_matrices(self, N_p=None, N_tilde=None,
                                           mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                                           A_pow_tilde=None, sparse=None, mat_ops=None):

        # X_tilde = Phi_x @ x(0) + Gamma_v @ v_tilde + Gamma_omega @ omega_tilde + Gamma_5

        gen_kwargs = dict(_disable_process_args=True, N_tilde=N_tilde,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde, mld_info_k0=mld_info_k0,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        state_input_evolution_struct = StructDict()
        state_input_evolution_struct['Phi_x_N_tilde'] = self._gen_phi_x(**gen_kwargs)
        state_input_evolution_struct['Gamma_v_N_tilde'] = self._gen_gamma_v(**gen_kwargs)
        state_input_evolution_struct['Gamma_omega_N_tilde'] = self._gen_gamma_omega(**gen_kwargs)
        state_input_evolution_struct['Gamma_5_N_tilde'] = self._gen_gamma_5(**gen_kwargs)

        self._update_with_N_p_slices(state_input_evolution_struct, mld_info_k0['n_states'], N_p)

        return state_input_evolution_struct

    @process_method_args_decor(_process_base_args, _process_mat_op_args, _process_A_pow_tilde_arg)
    def gen_output_evolution_matrices(self, N_p=None, N_tilde=None,
                                      mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                                      state_input_evolution_struct=None,
                                      A_pow_tilde=None, sparse=None, mat_ops=None):

        # Y_tilde = L_x @ x_0 + L_v @ v_tilde + L_omega @ omega_tilde + L_5

        gen_kwargs = dict(_disable_process_args=True, N_tilde=N_tilde,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde, mld_info_k0=mld_info_k0,
                          sparse=sparse, mat_ops=mat_ops)

        if state_input_evolution_struct is None:
            state_input_evolution_struct = (
                self.gen_state_input_evolution_matrices(N_p=N_p, A_pow_tilde=A_pow_tilde, **gen_kwargs))

        Phi_x_N_tilde, Gamma_v_N_tilde, Gamma_omega_N_tilde, Gamma_5_N_tilde = (
            state_input_evolution_struct.get_sub_list(
                ['Phi_x_N_tilde', 'Gamma_v_N_tilde', 'Gamma_omega_N_tilde', 'Gamma_5_N_tilde']
            ))

        output_evo_struct = StructDict()

        C_tilde = self._gen_C_tilde_diag(**gen_kwargs)
        D_v_tilde = self._gen_D_v_tilde_diag(**gen_kwargs)
        D4_tilde = self._gen_D4_tilde_diag(**gen_kwargs)
        d5_tilde = self._gen_d5_tilde(**gen_kwargs)

        output_evo_struct['L_x_N_tilde'] = C_tilde @ Phi_x_N_tilde
        output_evo_struct['L_v_N_tilde'] = C_tilde @ Gamma_v_N_tilde + D_v_tilde
        output_evo_struct['L_omega_N_tilde'] = C_tilde @ Gamma_omega_N_tilde + D4_tilde
        output_evo_struct['L_5_N_tilde'] = C_tilde @ Gamma_5_N_tilde + d5_tilde

        self._update_with_N_p_slices(output_evo_struct, mld_info_k0['n_outputs'], N_p)

        return output_evo_struct

    @process_method_args_decor(_process_base_args, _process_mat_op_args, _process_A_pow_tilde_arg)
    def gen_cons_evolution_matrices(self, N_p=None, N_tilde=None,
                                    mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                                    state_input_evolution_struct=None,
                                    output_evolution_struct=None,
                                    A_pow_tilde=None, sparse=None, mat_ops=None):

        # H_v @ v_tilde <= H_x @ x_0 + H_omega @ omega_tilde + H_5

        gen_kwargs = dict(_disable_process_args=True, N_tilde=N_tilde,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde, mld_info_k0=mld_info_k0,
                          sparse=sparse, mat_ops=mat_ops)

        if state_input_evolution_struct is None:
            state_input_evolution_struct = (
                self.gen_state_input_evolution_matrices(N_p=N_p, A_pow_tilde=A_pow_tilde,
                                                        **gen_kwargs))

        Phi_x_N_tilde, Gamma_v_N_tilde, Gamma_omega_N_tilde, Gamma_5_N_tilde = (
            state_input_evolution_struct.get_sub_list(
                ['Phi_x_N_tilde', 'Gamma_v_N_tilde', 'Gamma_omega_N_tilde', 'Gamma_5_N_tilde']
            ))

        if output_evolution_struct is None:
            output_evolution_struct = (
                self.gen_output_evolution_matrices(N_p=N_p, A_pow_tilde=A_pow_tilde,
                                                   state_input_evolution_struct=state_input_evolution_struct,
                                                   **gen_kwargs))

        L_x_N_tilde, L_v_N_tilde, L_omega_N_tilde, L_5_N_tilde = (
            output_evolution_struct.get_sub_list(
                ['L_x_N_tilde', 'L_v_N_tilde', 'L_omega_N_tilde', 'L_5_N_tilde']
            ))

        cons_evo_struct = StructDict()

        E_tilde = self._gen_E_tilde_diag(**gen_kwargs)
        F_v_tilde = self._gen_F_v_tilde_diag(**gen_kwargs)
        F4_tilde = self._gen_F4_tilde_diag(**gen_kwargs)
        f5_tilde = self._gen_f5_tilde(**gen_kwargs)
        G_tilde = self._gen_G_tilde_diag(**gen_kwargs)

        cons_evo_struct['H_x_N_tilde'] = - (E_tilde @ Phi_x_N_tilde + G_tilde @ L_x_N_tilde)
        cons_evo_struct['H_v_N_tilde'] = E_tilde @ Gamma_v_N_tilde + F_v_tilde + G_tilde @ L_v_N_tilde
        cons_evo_struct['H_omega_N_tilde'] = -(E_tilde @ Gamma_omega_N_tilde + F4_tilde + G_tilde @ L_omega_N_tilde)
        cons_evo_struct['H_5_N_tilde'] = f5_tilde - (E_tilde @ Gamma_5_N_tilde + G_tilde @ L_5_N_tilde)

        self._update_with_N_p_slices(cons_evo_struct, mld_info_k0['n_constraints'], N_p)

        return cons_evo_struct

    def _update_with_N_p_slices(self, current_struct, row_partition_size, N_p):
        for evo_mat_N_tilde_name, evo_mat_N_tilde in list(current_struct.items()):
            if evo_mat_N_tilde_name.endswith('N_tilde'):
                current_struct[evo_mat_N_tilde_name.replace('N_tilde', 'N_p')] = (
                    evo_mat_N_tilde[:N_p * row_partition_size, :])

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_A_pow_tilde(self, N_tilde=None, mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                         sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            mld_numeric_k0 = mld_numeric_tilde[0]
        else:
            mld_numeric_k0 = mld_numeric

        if 'A' in mld_numeric_k0._all_empty_mats:
            return tuple([mld_numeric_k0['A']]*N_tilde)

        # A_pow_tilde = [(A_k)^0, (A_k+1)^1, (A_k+2)^2, ..., (A_k+N_p)^(N_p)]
        if mld_numeric_tilde:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric_tilde[0].A.shape))] + (
                [mat_ops.vmatrix(mld_numeric_tilde[k].A) for k in range(N_tilde - 1)])
        else:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric.A.shape))] + [mat_ops.vmatrix(mld_numeric.A)] * (N_tilde - 1)

        return tuple(itertools.accumulate(A_tilde, lambda x, y: mat_ops.vmatrix(x @ y)))

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_phi_x(self, N_tilde=None, mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                   A_pow_tilde=None, sparse=None, mat_ops=None, copy=None, **kwargs):

        # Phi_x = [(A_k)^0; (A_k+1)^1; (A_k+2)^2; ... ;(A_k+N_p)^(N_p)]
        Phi_x = mat_ops.package.vstack(A_pow_tilde)
        return Phi_x

    @process_method_args_decor(_process_base_args, _process_mat_op_args, _process_A_pow_tilde_arg)
    def _gen_gamma_v(self, N_tilde=None,
                     mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                     A_pow_tilde=None,
                     sparse=None, mat_ops=None):
        # col = [[0s],(A_k)^0*[B1_k, B2_k, B3_k],..., (A_k+N_p-1)^(N_p-1)*[B1_k+N_p-1, B2_k+N_p-1, B3_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_v = toeplitz(col, row)

        input_mat_names = ['B1', 'B2', 'B3', '_zeros_Psi_state_input']
        Gamma_v = self._gen_input_evolution_mat(N_tilde=N_tilde,
                                                mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                                input_mat_names=input_mat_names,
                                                A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        return Gamma_v

    @process_method_args_decor(_process_base_args, _process_mat_op_args, _process_A_pow_tilde_arg)
    def _gen_gamma_omega(self, N_tilde=None,
                         mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                         A_pow_tilde=None,
                         sparse=None, mat_ops=None):

        # col = [[0s],(A_k)^0*[B4],..., (A_k+N_p-1)^(N_p-1)*[B4_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_omega = toeplitz(col, row)

        input_mat_names = ['B4']
        Gamma_omega = self._gen_input_evolution_mat(N_tilde=N_tilde,
                                                    mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                                    input_mat_names=input_mat_names,
                                                    A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)
        return Gamma_omega

    @process_method_args_decor(_process_base_args, _process_mat_op_args, _process_A_pow_tilde_arg)
    def _gen_gamma_5(self, N_tilde=None,
                     mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                     A_pow_tilde=None,
                     sparse=None, mat_ops=None):

        # col = [[0s],(A_k)^0*[b5],..., (A_k+N_p-1)^(N_p-1)*[b5_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_5 = toeplitz(col, row)
        input_mat_names = ['b5']
        Gamma_5_tilde = self._gen_input_evolution_mat(N_tilde=N_tilde,
                                                      mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                                      input_mat_names=input_mat_names,
                                                      A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        n, m = Gamma_5_tilde.shape
        return Gamma_5_tilde @ np.ones((m, 1))

    ### OUTPUT EVOLUTION MATRIX COMPONENTS ###

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_C_tilde_diag(self, N_tilde=None,
                          mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                          sparse=None, mat_ops=None):

        output_mat_names = ['C']

        C_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde,
                                           mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                           mat_names=output_mat_names,
                                           sparse=sparse, mat_ops=mat_ops)

        return C_tilde

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_D_v_tilde_diag(self, N_tilde=None,
                            mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                            sparse=None, mat_ops=None):

        output_mat_names = ['D1', 'D2', 'D3', '_zeros_Psi_output']

        D_123_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde,
                                               mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                               mat_names=output_mat_names,
                                               sparse=sparse, mat_ops=mat_ops)

        return D_123_tilde

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_D4_tilde_diag(self, N_tilde=None,
                           mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                           sparse=None, mat_ops=None):

        output_mat_names = ['D4']

        D4_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde,
                                            mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                            mat_names=output_mat_names,
                                            sparse=sparse, mat_ops=mat_ops)

        return D4_tilde

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_d5_tilde(self, N_tilde=None,
                      mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                      sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            d5_tilde = np.vstack([mld_numeric_tilde[k]['d5'] for k in range(N_tilde)])
        else:
            d5_tilde = np.repeat(mld_numeric['d5'], N_tilde, axis=0)

        return d5_tilde

    ### CONSTRAINT EVOLUTION MATRIX COMPONENTS ###

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_E_tilde_diag(self, N_tilde=None,
                          mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                          sparse=None, mat_ops=None):

        cons_mat_names = ['E']

        E_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde,
                                           mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                           mat_names=cons_mat_names,
                                           sparse=sparse, mat_ops=mat_ops)

        return E_tilde

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_F_v_tilde_diag(self, N_tilde=None,
                            mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                            sparse=None, mat_ops=None):

        cons_mat_names = ['F1', 'F2', 'F3', 'Psi']

        F_123_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde, mld_numeric=mld_numeric,
                                               mld_numeric_tilde=mld_numeric_tilde,
                                               mat_names=cons_mat_names,
                                               sparse=sparse, mat_ops=mat_ops)

        return F_123_tilde

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_F4_tilde_diag(self, N_tilde=None,
                           mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                           sparse=None, mat_ops=None):

        cons_mat_names = ['F4']

        F4_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde, mld_numeric=mld_numeric,
                                            mld_numeric_tilde=mld_numeric_tilde,
                                            mat_names=cons_mat_names,
                                            sparse=sparse, mat_ops=mat_ops)

        return F4_tilde

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_f5_tilde(self, N_tilde=None,
                      mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                      sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            f5_tilde = np.vstack([mld_numeric_tilde[k]['f5'] for k in range(N_tilde)])
        else:
            f5_tilde = np.repeat(mld_numeric['f5'], N_tilde, axis=0)

        return f5_tilde

    @process_method_args_decor(_process_base_args, _process_mat_op_args)
    def _gen_G_tilde_diag(self, N_tilde=None,
                          mld_numeric=None, mld_numeric_tilde=None, mld_info_k0=None,
                          sparse=None, mat_ops=None):

        cons_mat_names = ['G']

        G_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde, mld_numeric=mld_numeric,
                                           mld_numeric_tilde=mld_numeric_tilde,
                                           mat_names=cons_mat_names,
                                           sparse=sparse, mat_ops=mat_ops)

        return G_tilde

    @staticmethod
    def _gen_input_evolution_mat(N_tilde=None,
                                 mld_numeric=None, mld_numeric_tilde=None,
                                 input_mat_names=None, A_pow_tilde=None,
                                 sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            mld_numeric_k0 = mld_numeric_tilde[0]
        else:
            mld_numeric_k0 = mld_numeric

        non_empty_mats = [mat_name for mat_name in input_mat_names if mat_name not in mld_numeric_k0._all_empty_mats]
        non_zero_mats = [mat_name for mat_name in input_mat_names if mat_name not in mld_numeric_k0._all_zero_mats]

        B_hstack_k0 = mat_ops.vmatrix(mat_ops.package.hstack(
            [mat_ops.hmatrix(mld_numeric_k0[input_mat_name]) for input_mat_name in input_mat_names]))

        if non_zero_mats:
            if mld_numeric_tilde:
                B_hstack_tilde = [
                    mat_ops.vmatrix(mat_ops.package.hstack(
                        [mat_ops.hmatrix(mld_numeric_tilde[k][input_mat_name]) for input_mat_name in non_empty_mats])
                    ) for k in range(N_tilde)]

                col_list = ([mat_ops.zeros(B_hstack_tilde[0].shape)] +
                            [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack_tilde[k]) for k in range(N_tilde - 1)])
                row_list = [mat_ops.zeros(B_hstack_tilde[0].shape)] * (N_tilde)
            else:
                col_list = ([mat_ops.zeros(B_hstack_k0.shape)] +
                            [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack_k0) for k in range(N_tilde - 1)])
                row_list = [mat_ops.zeros(B_hstack_k0.shape)] * (N_tilde)

            return block_toeplitz(col_list, row_list, sparse=sparse)
        else:
            return mat_ops.zeros(tuple(np.array(B_hstack_k0.shape)*N_tilde))



    @staticmethod
    def _gen_mat_tilde_diag(N_tilde=None,
                            mld_numeric: MldModel = None, mld_numeric_tilde=None,
                            mat_names=None,
                            sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            mld_numeric_k0 = mld_numeric_tilde[0]
        else:
            mld_numeric_k0 = mld_numeric

        non_empty_mats = [mat_name for mat_name in mat_names if mat_name not in mld_numeric_k0._all_empty_mats]
        non_zero_mats = [mat_name for mat_name in mat_names if mat_name not in mld_numeric_k0._all_zero_mats]

        mat_hstack_k0 = mat_ops.vmatrix(mat_ops.package.hstack(
            [mat_ops.hmatrix(mld_numeric_k0[mat_name]) for mat_name in mat_names]))

        if non_zero_mats:  # constant non-zero matrices exist
            if mld_numeric_tilde:
                mat_hstack_tilde = [
                    mat_ops.vmatrix(mat_ops.package.hstack(
                        [mat_ops.hmatrix(mld_numeric_tilde[k][mat_name]) for mat_name in non_empty_mats])
                    ) for k in range(N_tilde)]

            else:
                mat_hstack_tilde = [mat_hstack_k0] * N_tilde

            return mat_ops.block_diag(mat_hstack_tilde)
        else:
            return mat_ops.zeros(tuple(np.array(mat_hstack_k0.shape)*N_tilde))


MpcOptVarStruct = struct_prop_fixed_dict('MpcOptVarStruct', ['var_name', 'var_mat_N_tilde', 'var_N_tilde', 'var_N_p'],
                                         sorted_repr=False)


class MpcOptimizationVars(MpcComponentsBase):
    _controllable_vars = MldInfo._controllable_var_names
    _concat_controllable_vars = MldInfo._concat_controllable_var_names
    _state_output_vars = MldInfo._state_var_names + MldInfo._output_var_names

    _field_names = _controllable_vars + _concat_controllable_vars + _state_output_vars
    _field_names_set = frozenset(_field_names)
    _var_names = _field_names

    __internal_names = ['mpc_evolution_mats']
    _internal_names_set = set(__internal_names)

    def __init__(self, mpc_controller: MpcController):
        super(MpcOptimizationVars, self).__init__(mpc_controller)
        self.update()

    def _reset(self):
        super(MpcOptimizationVars, self)._reset()
        self._base_dict_init({var_name: MpcOptVarStruct(var_name=var_name) for var_name in self._var_names})

    @property
    def mpc_evolution_mats(self):
        return self._mpc_controller._mpc_evo_mats

    @process_method_args_decor(_process_base_args)
    def update(self, N_p=None, N_tilde=None,
               mld_numeric: MldModel = None, mld_numeric_tilde=None,
               mld_info_k0: MldInfo = None):

        self.gen_optimization_vars(_disable_process_args=True, N_p=N_p, N_tilde=N_tilde,
                                   mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                   mld_info_k0=mld_info_k0)

        self.set_with_N_p = N_p
        self.set_with_N_tilde = N_tilde

    @process_method_args_decor(_process_base_args)
    def gen_optimization_vars(self, N_p=None, N_tilde=None,
                              mld_numeric: MldModel = None, mld_numeric_tilde=None,
                              mld_info_k0: MldInfo = None):

        V_opt_var_names = mld_info_k0._controllable_var_names
        slack_names = mld_info_k0._slack_var_names

        # extract variable matrix_types from mld mld_infos
        if mld_numeric_tilde:
            var_types_tilde_mats = {
                var_name: (
                    np.hstack(
                        [mld_numeric_tilde[k].mld_info.get_var_type(var_name) for k in range(N_tilde)]
                    ) if mld_info_k0.get_var_dim(var_name) else np.empty(
                        (0, mld_info_k0.get_var_dim(var_name)) * N_tilde, dtype=np.str)
                ) for var_name in self._controllable_vars
            }
        else:
            var_types_tilde_mats = {
                var_name: (
                    np.repeat(mld_numeric.mld_info.get_var_type(var_name), N_tilde, axis=1))
                for var_name in self._controllable_vars
            }

        def to_bin_index(type_mat):
            return (
                list(map(tuple, np.argwhere(type_mat == 'b').tolist())))

        # generate individual variable tilde mats
        opt_var_tilde_mats = {
            var_name: (
                cvx.Variable(var_type_mat.shape, boolean=to_bin_index(var_type_mat),
                             name="".join([var_name.capitalize(), '_tilde_mat_N_tilde']),
                             nonneg=(var_name in slack_names or None)) if var_type_mat.size
                else np.empty((0, N_tilde))
            ) for var_name, var_type_mat in var_types_tilde_mats.items()
        }

        # add combined input variable tilde mat
        opt_var_tilde_mats['v'] = cvx.vstack(
            [opt_var_tilde_mats[var_name] for var_name in V_opt_var_names if opt_var_tilde_mats[var_name].size])

        for var_name in itertools.chain(self._controllable_vars, self._concat_controllable_vars):
            var_mat_N_tilde = opt_var_tilde_mats[var_name]
            self[var_name]['var_mat_N_tilde'] = var_mat_N_tilde
            self[var_name]['var_N_tilde'] = var_N_tilde = (
                cvx.reshape(var_mat_N_tilde, (var_mat_N_tilde.size, 1)) if var_mat_N_tilde.size else np.empty(
                    (0, 1)))
            self[var_name]['var_N_p'] = var_N_tilde[:var_mat_N_tilde.shape[0] * N_p, :]

        # add state and output tilde mat
        state_output_N_tilde_vecs = {}
        if mld_info_k0.get_var_dim('x'):
            state_output_N_tilde_vecs['x'] = (
                    self.mpc_evolution_mats.state_input['Phi_x_N_tilde'] * self.x0 +
                    self.mpc_evolution_mats.state_input['Gamma_v_N_tilde'] @ self['v']['var_N_tilde'] +
                    self.mpc_evolution_mats.state_input['Gamma_omega_N_tilde'] @ self.omega_tilde_k +
                    self.mpc_evolution_mats.state_input['Gamma_5_N_tilde'])
        else:
            state_output_N_tilde_vecs['x'] = np.empty((0, 1))

        if mld_info_k0.get_var_dim('y'):
            state_output_N_tilde_vecs['y'] = (
                    (self.mpc_evolution_mats.output['L_x_N_tilde'] * self.x0 if mld_info_k0.get_var_dim(
                        'x') else np.zeros((mld_info_k0.get_var_dim('y') * N_tilde, 1))) +
                    self.mpc_evolution_mats.output['L_v_N_tilde'] @ self['v']['var_N_tilde'] +
                    self.mpc_evolution_mats.output['L_omega_N_tilde'] @ self.omega_tilde_k +
                    self.mpc_evolution_mats.output['L_5_N_tilde'])
        else:
            state_output_N_tilde_vecs['y'] = np.empty((0, 1))

        for var_name in self._state_output_vars:
            var_N_tilde = state_output_N_tilde_vecs[var_name]
            self[var_name]['var_N_tilde'] = var_N_tilde
            self[var_name]['var_N_p'] = var_N_tilde[:mld_info_k0.get_var_dim(var_name) * N_p, :]
            self[var_name]['var_mat_N_tilde'] = (
                cvx.reshape(var_N_tilde, (mld_info_k0.get_var_dim(var_name), N_tilde)) if var_N_tilde.size else
                np.empty((0, N_tilde)))


_mpc_objective_weight_types = ['linear', 'linear_f', 'quadratic', 'quadratic_f']
ObjectiveWeightStruct = struct_prop_fixed_dict('ObjectiveWeightStruct',
                                               _mpc_objective_weight_types,
                                               sorted_repr=False)


class MpcObjectiveWeights(MpcComponentsBase):
    _allowed_weight_types = set(_mpc_objective_weight_types)
    _field_names = MldInfo._var_names
    _field_names_set = frozenset(_field_names)

    _var_names = _field_names
    _default_dict = dict.fromkeys(_field_names)

    def _process_weight_names_args(self, f_kwargs=None, *,
                                   objective_weights_struct=ParNotSet, _var_kwargs=ParNotSet,
                                   **kwargs):
        _var_kwargs = _var_kwargs or {}
        if objective_weights_struct is None:
            objective_weights_struct = _var_kwargs
        else:
            objective_weights_struct.update(_var_kwargs)
        f_kwargs['objective_weights_struct'] = objective_weights_struct

    @process_method_args_decor(_process_weight_names_args)
    def __init__(self, mpc_controller: MpcController, objective_weights_struct=None, **kwargs):
        super(MpcObjectiveWeights, self).__init__(mpc_controller)
        self.update(_disable_process_args=True, objective_weights_struct=objective_weights_struct, **kwargs)

    def _reset(self):
        super(MpcObjectiveWeights, self)._reset()
        self._base_dict_init(self._default_dict)

    @property
    def opt_vars(self):
        return self._mpc_controller._opt_vars

    @process_method_args_decor(_process_weight_names_args)
    def update(self, objective_weights_struct=None, **kwargs):
        for var_name, weight in objective_weights_struct.items():
            if var_name in self._var_names:
                try:
                    for weight_type, value in weight.items():
                        self._set_weight(var_name, weight_type, value)
                except TypeError:
                    raise TypeError(f"{var_name} value in objective_weights_struct must be dictionary like")
            else:
                self._set_weight_from_weight_string(weight_name=var_name, value=weight)
        return self

    @process_method_args_decor(_process_weight_names_args)
    def set(self, objective_weights_struct=None, **kwargs):
        self.clear()
        self.update(_disable_process_args=True, objective_weights_struct=objective_weights_struct, **kwargs)

    def _set_weight_from_weight_string(self, weight_name, value):
        weight_type = (('linear' if weight_name[0].islower() else 'quadratic') +
                       ("_f" if weight_name.endswith('_f') else ""))
        var_name = weight_name.split('_')[1].lower()

        if var_name in self._var_names:
            self._set_weight(var_name, weight_type, value)
        else:
            raise ValueError(
                f"weight_name: '{weight_name}' is not valid. Must be of the form: [lower/upper]_[var_name] for stage "
                f"weights or [lower/upper]_[var_name]_f for terminal weights.")

    def _set_weight(self, var_name, weight_type, value):
        if value is not None:
            if weight_type not in self._allowed_weight_types:
                raise ValueError(
                    (f"Invalid weight type: '{weight_type}'. Weight type must be in {set(self._allowed_weight_types)}"))
            var_dim = self.mld_info_k0.get_var_dim(var_name)
            value = atleast_2d_col(value)
            value_shape = value.shape

            processed_value = self._weight_process_switcher[weight_type](self, weight_type=weight_type,
                                                                         var_name=var_name,
                                                                         var_dim=var_dim,
                                                                         value=value,
                                                                         value_shape=value_shape)
            if isinstance(self[var_name], ObjectiveWeightStruct):
                self[var_name][weight_type] = processed_value
            elif value is not None:
                self[var_name] = ObjectiveWeightStruct({weight_type: processed_value})
        else:
            return value

    def _process_linear_weight(self, weight_type, var_name, var_dim, value, value_shape, is_terminal):
        if value_shape[1] != 1:
            raise ValueError(f"Column dim of {weight_type} weight for opt_var: '{var_name}', must be 1.")
        elif is_terminal:
            if value_shape[0] == var_dim:
                return value
            else:
                raise ValueError(
                    f"Row dim of {weight_type} weight for opt_var: '{var_name}' must be in {{{var_dim}}}")
        elif value_shape[0] == (var_dim * self.N_p):
            return value
        # elif value_shape[0] == self.N_tilde:
        #      return value
        elif value_shape[0] == var_dim:
            return np.repeat(value, self.N_p, axis=0)
        else:
            raise ValueError(
                f"Row dim of {weight_type} weight for opt_var: '{var_name}', must be in {{{var_dim}, {var_dim}*N_p}}")

    def _process_quadratic_weight(self, weight_type, var_name, var_dim, value, value_shape, is_terminal):
        if value_shape[0] != value_shape[1]:
            raise ValueError(
                f"Quadratic weight for opt_var: '{var_name}', must be square. Currently has shape: {value_shape}")
        elif is_terminal:
            if value_shape[0] == var_dim:
                return value
            else:
                raise ValueError(
                    f"Row dim of {weight_type} weight for opt_var: '{var_name}' must be in {{{var_dim}}}")
        elif value_shape[0] == (var_dim * self.N_p):
            return value
        # elif value_shape[0] == self.N_tilde:
        #     return value
        elif value_shape[0] == var_dim:
            return block_diag_dense([value] * self.N_p)
        else:
            raise ValueError(
                f"Row dim of {weight_type} weight for opt_var: '{var_name}' must be in {{{var_dim}, "
                f"{var_dim}*N_p}}")

    def get_cost(self):
        cost = 0
        for var_name, weight in self.items():
            if weight is not None:
                for weight_type, value in weight.items():
                    if value is not None:
                        cost += self._apply_weight_to_var(var_name, weight_type, value)
        return cost

    def _apply_weight_to_var(self, var_name, weight_type, value):
        if value is not None:
            return self._weight_apply_switcher[weight_type](self, weight_type=weight_type,
                                                            var_name=var_name,
                                                            value=value)
        else:
            return 0

    def _apply_linear_weight_to_var(self, var_name, weight_type, value, is_terminal):
        if is_terminal:
            cost = 0
        else:
            cost = np.transpose(value) @ self.opt_vars[var_name]['var_N_p']

        return cost

    def _apply_quadratic_weight_to_var(self, var_name, weight_type, value, is_terminal):
        if is_terminal:
            cost = 0
        else:
            cost = cvx.QuadForm(self.opt_vars[var_name]['var_N_p'], value)
        return cost

    _weight_apply_switcher = {
        'linear'     : functools.partial(_apply_linear_weight_to_var, is_terminal=False),
        'linear_f'   : functools.partial(_apply_linear_weight_to_var, is_terminal=True),
        'quadratic'  : functools.partial(_apply_quadratic_weight_to_var, is_terminal=False),
        'quadratic_f': functools.partial(_apply_quadratic_weight_to_var, is_terminal=True)
    }

    _weight_process_switcher = {
        'linear'     : functools.partial(_process_linear_weight, is_terminal=False),
        'linear_f'   : functools.partial(_process_linear_weight, is_terminal=True),
        'quadratic'  : functools.partial(_process_quadratic_weight, is_terminal=False),
        'quadratic_f': functools.partial(_process_quadratic_weight, is_terminal=True)
    }
