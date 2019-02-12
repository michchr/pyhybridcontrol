import itertools
from collections import namedtuple

import cvxpy as cvx
import numpy as np

from abc import abstractmethod
from controllers.mpc_controller.mpc_utils import process_base_args, process_mat_op_args, process_A_pow_tilde_arg
from models.mld_model import MldModel, MldInfo
from utils.decorator_utils import process_method_args_decor
from utils.func_utils import ParNotSet
from utils.matrix_utils import block_toeplitz, atleast_2d_col, block_diag_dense, matmul
from structdict import StructPropFixedDictMixin, struct_prop_fixed_dict, StructDict


class MpcComponentsBase(StructPropFixedDictMixin, dict):
    _field_names = ()
    _field_names_set = frozenset()

    __internal_names = ['_mpc_controller', '_N_p', '_N_tilde', '_mld_numeric_k', '_mld_numeric_tilde', 'set_with_N_p',
                        'set_with_N_tilde']
    _internal_names_set = set(__internal_names)

    def __init__(self, mpc_controller=None, N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet):
        self._base_dict_init()
        self._mpc_controller = mpc_controller
        self._N_p = N_p
        self._N_tilde = N_tilde
        self._mld_numeric_tilde = mld_numeric_tilde
        self._mld_numeric_k = mld_numeric_k
        if mpc_controller is None:
            self._N_p = N_p if N_p is not ParNotSet else None
            self._N_tilde = N_tilde if N_tilde is not ParNotSet else None
            self._mld_numeric_tilde = mld_numeric_tilde if mld_numeric_tilde is not ParNotSet else None
            self._mld_numeric_k = mld_numeric_k if mld_numeric_k is not ParNotSet else None
            if None in (self._N_p, self._N_tilde, self._mld_numeric_k):
                raise ValueError("N_p, N_tilde and mld_numeric_k must all be set with non none value.")
        self._reset()

    def _reset(self):
        self.set_with_N_p = None
        self.set_with_N_tilde = None

    def _update_set_with(self, N_p, N_tilde):
        self.set_with_N_p = N_p
        self.set_with_N_tilde = N_tilde

    @property
    def N_p(self):
        N_p = self._N_p
        return (N_p if N_p is not ParNotSet else self._mpc_controller.N_p)

    @property
    def N_tilde(self):
        N_tilde = self._N_tilde
        return (N_tilde if N_tilde is not ParNotSet else self._mpc_controller.N_tilde)

    @property
    def mld_numeric_k(self) -> MldModel:
        mld_numeric_tilde = self._mld_numeric_tilde
        mld_numeric_k = mld_numeric_tilde[0] if mld_numeric_tilde not in (None, ParNotSet) else self._mld_numeric_k
        return (mld_numeric_k if mld_numeric_k is not ParNotSet else self._mpc_controller.mld_numeric_k)

    @property
    def mld_numeric_tilde(self):
        mld_numeric_tilde = self._mld_numeric_tilde
        return (mld_numeric_tilde if mld_numeric_tilde is not ParNotSet else self._mpc_controller.mld_numeric_tilde)

    @property
    def mld_info_k(self) -> MldInfo:
        return self.mld_numeric_k.mld_info

    @property
    def x_k(self):
        return self._mpc_controller._x_k

    @property
    def omega_tilde_k(self):
        return self._mpc_controller._omega_tilde_k


_mpc_evo_mat_types_names = ['state_input', 'output', 'constraint']
MpcEvoMatricesStruct = struct_prop_fixed_dict('MpcEvoMatricesStruct', _mpc_evo_mat_types_names)


class MpcSysEvoMatrices(MpcComponentsBase):
    _MldModelMatTypesNamedTup = namedtuple('matrix_types', _mpc_evo_mat_types_names)
    matrix_types = _MldModelMatTypesNamedTup._make(_mpc_evo_mat_types_names)

    _var_lengths = ['N_p', 'N_tilde']
    _state_input_evo_mats = ['Phi_x', 'Gamma_v', 'Gamma_omega', 'Gamma_5']
    _output_evo_mats = ['L_x', 'L_v', 'L_omega', 'L_5']
    _constraint_evo_mats = ['H_x', 'H_v', 'H_omega', 'H_5']

    _state_input_evo_mat_names = [name + '_' + lenght for name, lenght in
                                  itertools.product(_state_input_evo_mats, _var_lengths)]
    _output_evo_mat_names = [name + '_' + lenght for name, lenght in
                             itertools.product(_output_evo_mats, _var_lengths)]
    _constraint_evo_mat_names = [name + '_' + lenght for name, lenght in
                                 itertools.product(_constraint_evo_mats, _var_lengths)]

    StateInputEvoMatStruct = struct_prop_fixed_dict('StateInputEvoMatStruct', _state_input_evo_mat_names)
    OutputEvoMatStruct = struct_prop_fixed_dict('StateInputEvoMatStruct', _output_evo_mat_names)
    ConstraintEvoMatStruct = struct_prop_fixed_dict('ConstraintEvoMatStruct',
                                                    _constraint_evo_mat_names)

    _field_names = matrix_types
    _field_names_set = frozenset(_field_names)

    def _reset(self):
        super(MpcSysEvoMatrices, self)._reset()
        self._base_dict_init({
            self.matrix_types.state_input: self.StateInputEvoMatStruct(),
            self.matrix_types.output     : self.OutputEvoMatStruct(),
            self.matrix_types.constraint : self.ConstraintEvoMatStruct()
        })

    def __init__(self, mpc_controller=None, N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet):
        super(MpcSysEvoMatrices, self).__init__(mpc_controller, N_p=N_p, N_tilde=N_tilde,
                                                mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)
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

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def update(self, N_p=None, N_tilde=None,
               mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
               A_pow_tilde=None, sparse=None, mat_ops=None):

        self._base_dict_update(
            self.gen_mpc_evo_matrices(_disable_process_args=True, N_p=N_p, N_tilde=N_tilde,
                                      mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                      mld_info_k=mld_info_k,
                                      A_pow_tilde=A_pow_tilde,
                                      sparse=sparse, mat_ops=mat_ops))

        self._update_set_with(N_p, N_tilde)

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def gen_mpc_evo_matrices(self, N_p=None, N_tilde=None,
                             mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                             A_pow_tilde=None, sparse=None, mat_ops=None):

        gen_kwargs = dict(_disable_process_args=True, N_tilde=N_tilde,
                          mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde, mld_info_k=mld_info_k,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)
        mpc_evo_struct = MpcEvoMatricesStruct({self.matrix_types.state_input: self.StateInputEvoMatStruct(),
                                               self.matrix_types.output     : self.OutputEvoMatStruct(),
                                               self.matrix_types.constraint : self.ConstraintEvoMatStruct()}
                                              )

        mpc_evo_struct[self.matrix_types.state_input].update(
            self.gen_state_input_evo_matrices(N_p=N_p, **gen_kwargs))
        mpc_evo_struct[self.matrix_types.output].update(
            self.gen_output_evo_matrices(N_p=N_p,
                                         state_input_evo_struct=mpc_evo_struct[self.matrix_types.state_input],
                                         **gen_kwargs))

        mpc_evo_struct[self.matrix_types.constraint].update(
            self.gen_cons_evo_matrices(N_p=N_p,
                                       state_input_evo_struct=mpc_evo_struct[self.matrix_types.state_input],
                                       output_evo_struct=mpc_evo_struct[self.matrix_types.output],
                                       **gen_kwargs))

        return mpc_evo_struct

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def gen_state_input_evo_matrices(self, N_p=None, N_tilde=None,
                                     mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                                     A_pow_tilde=None, sparse=None, mat_ops=None):

        # X_tilde = Phi_x @ x(0) + Gamma_v @ v_tilde + Gamma_omega @ omega_tilde + Gamma_5

        gen_kwargs = dict(_disable_process_args=True, N_tilde=N_tilde,
                          mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde, mld_info_k=mld_info_k,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        state_input_evo_struct = self.StateInputEvoMatStruct()
        state_input_evo_struct['Phi_x_N_tilde'] = self._gen_phi_x(**gen_kwargs)
        state_input_evo_struct['Gamma_v_N_tilde'] = self._gen_gamma_v(**gen_kwargs)
        state_input_evo_struct['Gamma_omega_N_tilde'] = self._gen_gamma_omega(**gen_kwargs)
        state_input_evo_struct['Gamma_5_N_tilde'] = self._gen_gamma_5(**gen_kwargs)

        self._update_with_N_p_slices(state_input_evo_struct, mld_info_k['n_states'], N_p)

        return state_input_evo_struct

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def gen_output_evo_matrices(self, N_p=None, N_tilde=None,
                                mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                                state_input_evo_struct=None,
                                A_pow_tilde=None, sparse=None, mat_ops=None):

        # Y_tilde = L_x @ x_0 + L_v @ v_tilde + L_omega @ omega_tilde + L_5

        gen_kwargs = dict(_disable_process_args=True, N_tilde=N_tilde,
                          mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde, mld_info_k=mld_info_k,
                          sparse=sparse, mat_ops=mat_ops)

        if state_input_evo_struct is None:
            state_input_evo_struct = (
                self.gen_state_input_evo_matrices(N_p=N_p, A_pow_tilde=A_pow_tilde, **gen_kwargs))

        Phi_x_N_tilde, Gamma_v_N_tilde, Gamma_omega_N_tilde, Gamma_5_N_tilde = (
            state_input_evo_struct.get_sub_list(
                ['Phi_x_N_tilde', 'Gamma_v_N_tilde', 'Gamma_omega_N_tilde', 'Gamma_5_N_tilde']
            ))

        output_evo_struct = self.OutputEvoMatStruct()

        C_tilde = self._gen_C_tilde_diag(**gen_kwargs)
        D_v_tilde = self._gen_D_v_tilde_diag(**gen_kwargs)
        D4_tilde = self._gen_D4_tilde_diag(**gen_kwargs)
        d5_tilde = self._gen_d5_tilde(**gen_kwargs)

        output_evo_struct['L_x_N_tilde'] = C_tilde @ Phi_x_N_tilde
        output_evo_struct['L_v_N_tilde'] = C_tilde @ Gamma_v_N_tilde + D_v_tilde
        output_evo_struct['L_omega_N_tilde'] = C_tilde @ Gamma_omega_N_tilde + D4_tilde
        output_evo_struct['L_5_N_tilde'] = C_tilde @ Gamma_5_N_tilde + d5_tilde

        self._update_with_N_p_slices(output_evo_struct, mld_info_k['n_outputs'], N_p)

        return output_evo_struct

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def gen_cons_evo_matrices(self, N_p=None, N_tilde=None,
                              mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                              state_input_evo_struct=None,
                              output_evo_struct=None,
                              A_pow_tilde=None, sparse=None, mat_ops=None):

        # H_v @ v_tilde <= H_x @ x_0 + H_omega @ omega_tilde + H_5

        gen_kwargs = dict(_disable_process_args=True, N_tilde=N_tilde,
                          mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde, mld_info_k=mld_info_k,
                          sparse=sparse, mat_ops=mat_ops)

        if state_input_evo_struct is None:
            state_input_evo_struct = (
                self.gen_state_input_evo_matrices(N_p=N_p, A_pow_tilde=A_pow_tilde,
                                                  **gen_kwargs))

        Phi_x_N_tilde, Gamma_v_N_tilde, Gamma_omega_N_tilde, Gamma_5_N_tilde = (
            state_input_evo_struct.get_sub_list(
                ['Phi_x_N_tilde', 'Gamma_v_N_tilde', 'Gamma_omega_N_tilde', 'Gamma_5_N_tilde']
            ))

        if output_evo_struct is None:
            output_evo_struct = (
                self.gen_output_evo_matrices(N_p=N_p, A_pow_tilde=A_pow_tilde,
                                             state_input_evo_struct=state_input_evo_struct,
                                             **gen_kwargs))

        L_x_N_tilde, L_v_N_tilde, L_omega_N_tilde, L_5_N_tilde = (
            output_evo_struct.get_sub_list(
                ['L_x_N_tilde', 'L_v_N_tilde', 'L_omega_N_tilde', 'L_5_N_tilde']
            ))

        cons_evo_struct = self.ConstraintEvoMatStruct()

        E_tilde = self._gen_E_tilde_diag(**gen_kwargs)
        F_v_tilde = self._gen_F_v_tilde_diag(**gen_kwargs)
        F4_tilde = self._gen_F4_tilde_diag(**gen_kwargs)
        f5_tilde = self._gen_f5_tilde(**gen_kwargs)
        G_tilde = self._gen_G_tilde_diag(**gen_kwargs)

        cons_evo_struct['H_x_N_tilde'] = -(E_tilde @ Phi_x_N_tilde + G_tilde @ L_x_N_tilde)
        cons_evo_struct['H_v_N_tilde'] = E_tilde @ Gamma_v_N_tilde + F_v_tilde + G_tilde @ L_v_N_tilde
        cons_evo_struct['H_omega_N_tilde'] = -(E_tilde @ Gamma_omega_N_tilde + F4_tilde + G_tilde @ L_omega_N_tilde)
        cons_evo_struct['H_5_N_tilde'] = f5_tilde - (E_tilde @ Gamma_5_N_tilde + G_tilde @ L_5_N_tilde)

        self._update_with_N_p_slices(cons_evo_struct, mld_info_k['n_constraints'], N_p)

        return cons_evo_struct

    def _update_with_N_p_slices(self, current_struct, row_partition_size, N_p):
        for evo_mat_N_tilde_name, evo_mat_N_tilde in list(current_struct.items()):
            if evo_mat_N_tilde_name.endswith('N_tilde'):
                current_struct[evo_mat_N_tilde_name.replace('N_tilde', 'N_p')] = (
                    evo_mat_N_tilde[:N_p * row_partition_size, :])

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_A_pow_tilde(self, N_tilde=None, mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                         sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            mld_numeric_k0 = mld_numeric_tilde[0]
        else:
            mld_numeric_k0 = mld_numeric_k

        if 'A' in mld_numeric_k0._all_empty_mats:
            return tuple([mat_ops.vmatrix(mld_numeric_k0['A'])] * N_tilde)

        # A_pow_tilde = [(A_k)^0, (A_k+1)^1, (A_k+2)^2, ..., (A_k+N_p)^(N_p)]
        if mld_numeric_tilde:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric_tilde[0].A.shape))] + (
                [mat_ops.vmatrix(mld_numeric_tilde[k].A) for k in range(N_tilde - 1)])
        else:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric_k.A.shape))] + [mat_ops.vmatrix(mld_numeric_k.A)] * (
                    N_tilde - 1)

        return tuple(itertools.accumulate(A_tilde, lambda x, y: mat_ops.vmatrix(x @ y)))

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def _gen_phi_x(self, N_tilde=None, mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                   A_pow_tilde=None, sparse=None, mat_ops=None, **kwargs):

        # Phi_x = [(A_k)^0; (A_k+1)^1; (A_k+2)^2; ... ;(A_k+N_p)^(N_p)]
        Phi_x = mat_ops.package.vstack(A_pow_tilde)
        return Phi_x

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def _gen_gamma_v(self, N_tilde=None,
                     mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                     A_pow_tilde=None,
                     sparse=None, mat_ops=None):
        # col = [[0s],(A_k)^0*[B1_k, B2_k, B3_k],..., (A_k+N_p-1)^(N_p-1)*[B1_k+N_p-1, B2_k+N_p-1, B3_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_v = toeplitz(col, row)

        input_mat_names = ['B1', 'B2', 'B3', '_zeros_Psi_state_input']
        Gamma_v = self._gen_input_evo_mat(N_tilde=N_tilde,
                                          mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                          input_mat_names=input_mat_names,
                                          A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        return Gamma_v

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def _gen_gamma_omega(self, N_tilde=None,
                         mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                         A_pow_tilde=None,
                         sparse=None, mat_ops=None):

        # col = [[0s],(A_k)^0*[B4],..., (A_k+N_p-1)^(N_p-1)*[B4_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_omega = toeplitz(col, row)

        input_mat_names = ['B4']
        Gamma_omega = self._gen_input_evo_mat(N_tilde=N_tilde,
                                              mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                              input_mat_names=input_mat_names,
                                              A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)
        return Gamma_omega

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def _gen_gamma_5(self, N_tilde=None,
                     mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                     A_pow_tilde=None,
                     sparse=None, mat_ops=None):

        # col = [[0s],(A_k)^0*[b5],..., (A_k+N_p-1)^(N_p-1)*[b5_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_5 = toeplitz(col, row)
        input_mat_names = ['b5']
        Gamma_5_tilde = self._gen_input_evo_mat(N_tilde=N_tilde,
                                                mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                                input_mat_names=input_mat_names,
                                                A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        n, m = Gamma_5_tilde.shape
        return Gamma_5_tilde @ np.ones((m, 1))

    ### OUTPUT EVOLUTION MATRIX COMPONENTS ###

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_C_tilde_diag(self, N_tilde=None,
                          mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                          sparse=None, mat_ops=None):

        output_mat_names = ['C']

        C_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde,
                                           mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                           mat_names=output_mat_names,
                                           sparse=sparse, mat_ops=mat_ops)

        return C_tilde

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_D_v_tilde_diag(self, N_tilde=None,
                            mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                            sparse=None, mat_ops=None):

        output_mat_names = ['D1', 'D2', 'D3', '_zeros_Psi_output']

        D_123_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde,
                                               mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                               mat_names=output_mat_names,
                                               sparse=sparse, mat_ops=mat_ops)

        return D_123_tilde

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_D4_tilde_diag(self, N_tilde=None,
                           mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                           sparse=None, mat_ops=None):

        output_mat_names = ['D4']

        D4_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde,
                                            mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                            mat_names=output_mat_names,
                                            sparse=sparse, mat_ops=mat_ops)

        return D4_tilde

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_d5_tilde(self, N_tilde=None,
                      mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                      sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            d5_tilde = np.vstack([mld_numeric_tilde[k]['d5'] for k in range(N_tilde)])
        else:
            d5_tilde = np.tile(mld_numeric_k['d5'], (N_tilde, 1))

        return d5_tilde

    ### CONSTRAINT EVOLUTION MATRIX COMPONENTS ###

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_E_tilde_diag(self, N_tilde=None,
                          mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                          sparse=None, mat_ops=None):

        cons_mat_names = ['E']

        E_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde,
                                           mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                           mat_names=cons_mat_names,
                                           sparse=sparse, mat_ops=mat_ops)

        return E_tilde

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_F_v_tilde_diag(self, N_tilde=None,
                            mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                            sparse=None, mat_ops=None):

        cons_mat_names = ['F1', 'F2', 'F3', 'Psi']

        F_123_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde, mld_numeric_k=mld_numeric_k,
                                               mld_numeric_tilde=mld_numeric_tilde,
                                               mat_names=cons_mat_names,
                                               sparse=sparse, mat_ops=mat_ops)

        return F_123_tilde

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_F4_tilde_diag(self, N_tilde=None,
                           mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                           sparse=None, mat_ops=None):

        cons_mat_names = ['F4']

        F4_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde, mld_numeric_k=mld_numeric_k,
                                            mld_numeric_tilde=mld_numeric_tilde,
                                            mat_names=cons_mat_names,
                                            sparse=sparse, mat_ops=mat_ops)

        return F4_tilde

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_f5_tilde(self, N_tilde=None,
                      mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                      sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            f5_tilde = np.vstack([mld_numeric_tilde[k]['f5'] for k in range(N_tilde)])
        else:
            f5_tilde = np.tile(mld_numeric_k['f5'], (N_tilde, 1))

        return f5_tilde

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _test(self, N_tilde=None,
              mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
              sparse=None, mat_ops=None):
        pass

    @process_method_args_decor(process_base_args, process_mat_op_args)
    def _gen_G_tilde_diag(self, N_tilde=None,
                          mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                          sparse=None, mat_ops=None):

        cons_mat_names = ['G']

        G_tilde = self._gen_mat_tilde_diag(N_tilde=N_tilde, mld_numeric_k=mld_numeric_k,
                                           mld_numeric_tilde=mld_numeric_tilde,
                                           mat_names=cons_mat_names,
                                           sparse=sparse, mat_ops=mat_ops)

        return G_tilde

    @staticmethod
    def _gen_input_evo_mat(N_tilde=None,
                           mld_numeric_k=None, mld_numeric_tilde=None,
                           input_mat_names=None, A_pow_tilde=None,
                           sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            mld_numeric_k0 = mld_numeric_tilde[0]
        else:
            mld_numeric_k0 = mld_numeric_k

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
                            [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack_tilde[k])
                             for k in range(N_tilde - 1)])
                row_list = [mat_ops.zeros(B_hstack_tilde[0].shape)] * (N_tilde)
            else:
                col_list = ([mat_ops.zeros(B_hstack_k0.shape)] +
                            [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack_k0) for k in range(N_tilde - 1)])
                row_list = [mat_ops.zeros(B_hstack_k0.shape)] * (N_tilde)

            return block_toeplitz(col_list, row_list, sparse=sparse)
        else:
            return mat_ops.zeros(tuple(np.array(B_hstack_k0.shape) * N_tilde))

    @staticmethod
    def _gen_mat_tilde_diag(N_tilde=None,
                            mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
                            mat_names=None,
                            sparse=None, mat_ops=None):

        if mld_numeric_tilde:
            mld_numeric_k0 = mld_numeric_tilde[0]
        else:
            mld_numeric_k0 = mld_numeric_k

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
            return mat_ops.zeros(tuple(np.array(mat_hstack_k0.shape) * N_tilde))


MpcOptVarStruct = struct_prop_fixed_dict('MpcOptVarStruct', ['var_name', 'var_mat_N_tilde', 'var_N_tilde', 'var_N_p'],
                                         sorted_repr=False)


class MpcOptimizationVars(MpcComponentsBase):
    _controllable_vars = MldInfo._controllable_var_names
    _concat_controllable_vars = MldInfo._concat_controllable_var_names
    _state_output_vars = MldInfo._state_var_names + MldInfo._output_var_names

    _field_names = _controllable_vars + _concat_controllable_vars + _state_output_vars
    _field_names_set = frozenset(_field_names)
    _var_names = _field_names

    __internal_names = ['mpc_evo_mats']
    _internal_names_set = set(__internal_names)

    def __init__(self, mpc_controller=None, N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet):
        super(MpcOptimizationVars, self).__init__(mpc_controller, N_p=N_p, N_tilde=N_tilde,
                                                  mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)
        self.update()

    def _reset(self):
        super(MpcOptimizationVars, self)._reset()
        self._base_dict_init({var_name: MpcOptVarStruct(var_name=var_name) for var_name in self._var_names})

    @property
    def mpc_evo_mats(self):
        return self._mpc_controller._sys_evo_matrices

    @process_method_args_decor(process_base_args)
    def update(self, N_p=None, N_tilde=None,
               mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
               mld_info_k: MldInfo = None):

        self.gen_optimization_vars(_disable_process_args=True, N_p=N_p, N_tilde=N_tilde,
                                   mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                   mld_info_k=mld_info_k)

        self._update_set_with(N_p, N_tilde)

    @process_method_args_decor(process_base_args)
    def gen_optimization_vars(self, N_p=None, N_tilde=None,
                              mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
                              mld_info_k: MldInfo = None):

        V_opt_var_names = mld_info_k._controllable_var_names
        slack_names = mld_info_k._slack_var_names

        # extract variable matrix_types from mld mld_infos
        if mld_numeric_tilde:
            var_types_tilde_mats = {
                var_name: (
                    np.hstack(
                        [mld_numeric_tilde[k].mld_info.get_var_type(var_name) for k in range(N_tilde)]
                    )
                    if mld_info_k.get_var_dim(var_name) else (
                        np.empty((0, mld_info_k.get_var_dim(var_name)) * N_tilde, dtype=np.str))
                ) for var_name in self._controllable_vars
            }
        else:
            var_types_tilde_mats = {
                var_name: (
                    np.tile(mld_numeric_k.mld_info.get_var_type(var_name), (1, N_tilde)))
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

        state_output_vars = (
            self._gen_state_output_vars(_disable_process_args=True,
                                        x_k=self.x_k, omega_tilde_k=self.omega_tilde_k,
                                        N_p=N_p, N_tilde=N_tilde,
                                        mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                        mld_info_k=mld_info_k))
        self._base_dict_update(state_output_vars)

    @process_method_args_decor(process_base_args)
    def _gen_state_output_vars(self, x_k=None, omega_tilde_k=None,
                               N_p=None, N_tilde=None,
                               mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
                               mld_info_k: MldInfo = None):

        state_output_vars = StructDict(
            x=MpcOptVarStruct(var_name='x'),
            y=MpcOptVarStruct(var_name='y')
        )

        x_k = x_k if x_k is not None else self.x_k
        omega_tilde_k = omega_tilde_k if omega_tilde_k is not None else self.omega_tilde_k

        if mld_info_k.get_var_dim('x'):
            state_output_vars['x']['var_N_tilde'] = (
                    matmul(self.mpc_evo_mats.state_input['Phi_x_N_tilde'], x_k) +
                    matmul(self.mpc_evo_mats.state_input['Gamma_v_N_tilde'], self['v']['var_N_tilde']) +
                    matmul(self.mpc_evo_mats.state_input['Gamma_omega_N_tilde'], omega_tilde_k) +
                    self.mpc_evo_mats.state_input['Gamma_5_N_tilde']
            )
        else:
            state_output_vars['x']['var_N_tilde'] = np.empty((0, 1))

        if mld_info_k.get_var_dim('y'):
            state_output_vars['y']['var_N_tilde'] = (
                    matmul(self.mpc_evo_mats.output['L_x_N_tilde'], x_k) +
                    matmul(self.mpc_evo_mats.output['L_v_N_tilde'], self['v']['var_N_tilde']) +
                    matmul(self.mpc_evo_mats.output['L_omega_N_tilde'], omega_tilde_k) +
                    self.mpc_evo_mats.output['L_5_N_tilde']
            )
        else:
            state_output_vars['y']['var_N_tilde'] = np.empty((0, 1))

        for var_name in self._state_output_vars:
            var_N_tilde = state_output_vars[var_name]['var_N_tilde']
            state_output_vars[var_name]['var_N_tilde'] = var_N_tilde
            state_output_vars[var_name]['var_N_p'] = var_N_tilde[:mld_info_k.get_var_dim(var_name) * N_p, :]
            state_output_vars[var_name]['var_mat_N_tilde'] = (
                cvx.reshape(var_N_tilde, (mld_info_k.get_var_dim(var_name), N_tilde)) if var_N_tilde.size else
                np.empty((0, N_tilde)))

        return state_output_vars

    def get_opt_vars_with(self, x_k=None, omega_tilde_k=None):
        opt_vars = self.copy()
        state_output_vars = self._gen_state_output_vars(x_k=x_k, omega_tilde_k=omega_tilde_k)
        opt_vars._base_dict_update(state_output_vars)
        return opt_vars


class MpcObjectiveWeightBase(MpcComponentsBase):
    _field_names = ['weight_N_tilde', 'weight_N_p', 'weight_f']
    _field_names_set = frozenset(_field_names)
    _weight_type = 'base'
    __internal_names = ['_var_name', '_var_dim']

    def __init__(self, mpc_controller=None, N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet,
                 var_name=None, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        super(MpcObjectiveWeightBase, self).__init__(mpc_controller, N_p=N_p, N_tilde=N_tilde,
                                                     mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)

        self.update(var_name=var_name, weight_N_tilde=weight_N_tilde, weight_N_p=weight_N_p, weight_f=weight_f)

    @property
    def weight_type(self):
        return self._weight_type

    def update(self, var_name=None, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        if var_name is not None:
            self._var_name = var_name
        self._var_dim = self.mld_info_k.get_var_dim(self._var_name)
        self._set_weight(weight_N_tilde=weight_N_tilde, weight_N_p=weight_N_p, weight_f=weight_f)
        self._update_set_with(self.N_p, self.N_tilde)

    def set(self, var_name=None, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        self.clear()
        self.update(var_name=var_name, weight_N_tilde=weight_N_tilde, weight_N_p=weight_N_p, weight_f=weight_f)

    def _reset(self):
        super(MpcObjectiveWeightBase, self)._reset()
        self._base_dict_init(dict.fromkeys(self._field_names))

    @abstractmethod
    def _set_weight(self, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        pass


class MpcLinearWeight(MpcObjectiveWeightBase):
    _weight_type = 'linear'

    def _set_weight(self, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        weight = (self['weight_N_tilde'] if self['weight_N_tilde'] is not None else np.zeros(
            (self.N_tilde * self._var_dim, 1)))
        if weight_N_tilde is not None:
            weight[:] = self._process_linear_weight(value=weight_N_tilde, length=self.N_tilde,
                                                    length_name='N_tilde', is_terminal=False)
        elif weight_N_p is not None:
            weight[:self.N_p * self._var_dim, :1] = self._process_linear_weight(value=weight_N_p, length=self.N_p,
                                                                                length_name='N_p', is_terminal=False)
        if weight_f is not None:
            weight[-self._var_dim:, :1] = self._process_linear_weight(value=weight_f, length=1,
                                                                      length_name='1', is_terminal=True)

        self._base_dict_update(weight_N_tilde=weight,
                               weight_N_p=weight[:self.N_p * self._var_dim, :1],
                               weight_f=weight[-self._var_dim:, :1])

    def _process_linear_weight(self, value, length, length_name, is_terminal=False):
        var_dim = self._var_dim
        value = atleast_2d_col(value)
        value_shape = value.shape
        if value_shape[1] != 1:
            raise ValueError(f"Column dim of {self._weight_type} weight for opt_var: '{self._var_name}', must be 1.")
        elif is_terminal:
            if value_shape[0] == var_dim:
                return value
            else:
                raise ValueError(
                    f"Row dim of {self._weight_type} terminal weight for opt_var: '{self._var_name}' must "
                    f"be in {{{var_dim}}}")
        elif value_shape[0] == (var_dim * length):
            return value
        elif value_shape[0] == var_dim:
            return np.tile(value, (length, 1))
        else:
            raise ValueError(
                f"Row dim of {self._weight_type} weight for opt_var: '{self._var_name}', must be in "
                f"{{{var_dim}, {var_dim}*{length_name}}}")


class MpcQuadraticWeight(MpcObjectiveWeightBase):
    _weight_type = 'quadratic'

    def _set_weight(self, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        weight = (self['weight_N_tilde'] if self['weight_N_tilde'] is not None else np.zeros(
            (self.N_tilde * self._var_dim, self.N_tilde * self._var_dim)))
        if weight_N_tilde is not None:
            weight[:] = self._process_quadratic_weight(value=weight_N_tilde, length=self.N_tilde,
                                                       length_name='N_tilde', is_terminal=False)
        elif weight_N_p is not None:
            weight[:self.N_p * self._var_dim, :self.N_p * self._var_dim] = (
                self._process_quadratic_weight(value=weight_N_p, length=self.N_p,
                                               length_name='N_p', is_terminal=False))
        if weight_f is not None:
            weight[-self._var_dim:, -self._var_dim:] = self._process_quadratic_weight(value=weight_f, length=1,
                                                                                      length_name='1', is_terminal=True)

        self._base_dict_update(weight_N_tilde=weight,
                               weight_N_p=weight[:self.N_p * self._var_dim, :self.N_p * self._var_dim],
                               weight_f=weight[-self._var_dim:, -self._var_dim:])

    def _process_quadratic_weight(self, value, length, length_name, is_terminal=False):
        var_dim = self._var_dim
        value = atleast_2d_col(value)
        value_shape = value.shape
        if value_shape[0] != value_shape[1]:
            raise ValueError(
                f"Quadratic weight for opt_var: '{self._var_name}', must be square. Currently has shape: {value_shape}")
        elif is_terminal:
            if value_shape[0] == var_dim:
                return value
            else:
                raise ValueError(
                    f"Row dim of {self._weight_type} terminal weight for opt_var: '{self._var_name}' must "
                    f"be in {{{var_dim}}}")
        elif value_shape[0] == (var_dim * length):
            return value
        elif value_shape[0] == var_dim:
            return block_diag_dense([value] * length)
        else:
            raise ValueError(
                f"Row dim of {self._weight_type} weight for opt_var: '{self._var_name}', must be in "
                f"{{{var_dim}, {var_dim}*{length_name}}}")


_mpc_objective_weight_types = ['linear', 'quadratic']
ObjectiveWeightStruct = struct_prop_fixed_dict('ObjectiveWeightStruct',
                                               _mpc_objective_weight_types,
                                               sorted_repr=False)


class MpcObjectiveWeights(MpcComponentsBase):
    _allowed_weight_types = _mpc_objective_weight_types
    _allowed_post_fix = {'N_p', "N_tilde", "f", ""}

    _field_names = MldInfo._var_names
    _field_names_set = frozenset(_field_names)
    _var_names = _field_names

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
    def __init__(self, mpc_controller=None, objective_weights_struct=None, N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet, **kwargs):
        super(MpcObjectiveWeights, self).__init__(mpc_controller, N_p=N_p, N_tilde=N_tilde,
                                                  mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)
        self.update(_disable_process_args=True, objective_weights_struct=objective_weights_struct, **kwargs)

    def _reset(self):
        super(MpcObjectiveWeights, self)._reset()
        self._base_dict_init(self.base_dict.fromkeys(self._var_names))

    @process_method_args_decor(_process_weight_names_args)
    def update(self, objective_weights_struct=None, **kwargs):
        N_p = self.N_p
        N_tilde = self.N_tilde
        mld_info_k = self.mld_info_k
        update_weights = {}
        for var_name_or_weight_str, weight in objective_weights_struct.items():
            if var_name_or_weight_str in self._var_names:
                if isinstance(weight, MpcObjectiveWeightBase):
                    self._set_weight(update_weights, var_name_or_weight_str, weight.weight_type,
                                     weight['weight_N_tilde'], 'weight_N_tilde')
                else:
                    raise TypeError(
                        f"{var_name_or_weight_str} value in objective_weights_struct must be"
                        f" subclass of {MpcComponentsBase.__name__}")
            else:
                self._set_weight_from_weight_string(update_weights=update_weights, weight_name=var_name_or_weight_str,
                                                    value=weight, N_p=N_p, N_tilde=N_tilde, mld_info_k=mld_info_k)

        for var_name, weights in update_weights.items():
            if self.get(var_name) is None:
                self[var_name] = ObjectiveWeightStruct()
            if weights['linear']:
                if self[var_name]['linear'] is not None:
                    self[var_name]['linear'].update(var_name=var_name, **weights['linear'])
                else:
                    self[var_name]['linear'] = MpcLinearWeight(mpc_controller=self._mpc_controller,
                                                               N_p=self._N_p, N_tilde=self._N_tilde,
                                                               mld_numeric_k=self._mld_numeric_k,
                                                               mld_numeric_tilde=self._mld_numeric_tilde,
                                                               var_name=var_name, **weights['linear']
                                                               )
            if weights['quadratic']:
                if self[var_name]['quadratic'] is not None:
                    self[var_name]['quadratic'].update(var_name=var_name, **weights['quadratic'])
                else:
                    self[var_name]['quadratic'] = MpcQuadraticWeight(mpc_controller=self._mpc_controller,
                                                                     N_p=self._N_p, N_tilde=self._N_tilde,
                                                                     mld_numeric_k=self._mld_numeric_k,
                                                                     mld_numeric_tilde=self._mld_numeric_tilde,
                                                                     var_name=var_name, **weights['quadratic']
                                                                     )
        self._update_set_with(N_p, N_tilde)

    @process_method_args_decor(_process_weight_names_args)
    def set(self, objective_weights_struct=None, **kwargs):
        self.clear()
        self.update(_disable_process_args=True, objective_weights_struct=objective_weights_struct, **kwargs)

    def _set_weight_from_weight_string(self, update_weights, weight_name, value, N_p, N_tilde, mld_info_k):
        var_info = weight_name.split('_', 2)
        var_name = "".join(var_info[1:2]).lower()
        post_fix = "".join(var_info[2:3])
        if var_name in self._var_names and post_fix in self._allowed_post_fix:
            if value is None:
                return
            weight_type = 'linear' if var_info[0].islower() else 'quadratic'
            value = atleast_2d_col(value)
            var_dim = mld_info_k.get_var_dim(var_name)

            if post_fix:
                weight_length_name = "_".join(['weight', post_fix])
            elif value.shape[0] == var_dim or value.shape[0] == var_dim * N_tilde:
                weight_length_name = "weight_N_tilde"
            else:
                weight_length_name = "weight_N_p"

            self._set_weight(update_weights, var_name, weight_type, value, weight_length_name)
        else:
            raise ValueError(
                f"weight_name: '{weight_name}' is not valid. Must be of the form: [lower/upper]_[var_name] for "
                f"stage weights or [lower/upper]_[var_name]_f for terminal weights.")

    def _set_weight(self, update_weights, var_name, weight_type, value, weight_length_name):
        if not update_weights.get(var_name):
            update_weights[var_name] = (
                ObjectiveWeightStruct.fromkeys_withfunc(self._allowed_weight_types, lambda k: {}))
        update_weights[var_name][weight_type][weight_length_name] = value

    def gen_cost(self, opt_vars: MpcOptimizationVars):
        cost = 0
        for var_name, weights in self.items():
            if weights is not None:
                for weight_type, weight in weights.items():
                    if weight is not None:
                        value = weight['weight_N_tilde']
                    else:
                        continue
                    if value is not None:
                        opt_var = opt_vars.get(var_name)
                        if opt_var is not None:
                            cost += self._apply_weight_to_var(opt_var, weight_type, value)
                        else:
                            raise ValueError(f"opt_vars missing var_name:'{var_name}'")
        return cost

    def _apply_weight_to_var(self, opt_var, weight_type, value):
        if value is not None:
            return self._weight_apply_switcher[weight_type](self, opt_var=opt_var,
                                                            value=value)
        else:
            return 0

    def _apply_linear_weight_to_var(self, opt_var, value):
        cost = np.transpose(value) @ opt_var['var_N_tilde'][:value.shape[0]]

        return cost

    def _apply_quadratic_weight_to_var(self, opt_var, value):
        cost = cvx.QuadForm(opt_var['var_N_p'][:value.shape[1]], value)
        return cost

    _weight_apply_switcher = {
        'linear'   : _apply_linear_weight_to_var,
        'quadratic': _apply_quadratic_weight_to_var
    }
