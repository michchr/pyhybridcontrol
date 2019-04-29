import itertools
from collections import namedtuple

import numpy as np

from controllers.components.component_base import ComponentBase
from controllers.controller_utils import process_base_args, process_mat_op_args, process_A_pow_tilde_arg
from models.mld_model import MldModel, MldInfo
from structdict import named_fixed_struct_dict
from utils.decorator_utils import process_method_args_decor
from utils.func_utils import ParNotSet
from utils.matrix_utils import block_toeplitz
from utils.versioning import increments_version_decor

_mld_evo_mat_types_names = ['state_input', 'output', 'constraint']
MldEvoMatricesStruct = named_fixed_struct_dict('MldEvoMatricesStruct', _mld_evo_mat_types_names)


class MldEvoMatrices(ComponentBase):
    _MldModelMatTypesNamedTup = namedtuple('matrix_types', _mld_evo_mat_types_names)
    matrix_types = _MldModelMatTypesNamedTup._make(_mld_evo_mat_types_names)

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

    StateInputEvoMatStruct = named_fixed_struct_dict('StateInputEvoMatStruct', _state_input_evo_mat_names)
    OutputEvoMatStruct = named_fixed_struct_dict('OutputEvoMatStruct', _output_evo_mat_names)
    ConstraintEvoMatStruct = named_fixed_struct_dict('ConstraintEvoMatStruct',
                                                     _constraint_evo_mat_names)

    _field_names = matrix_types
    _field_names_set = frozenset(_field_names)

    def _reset(self):
        super(MldEvoMatrices, self)._reset()
        self._base_dict_init({
            self.matrix_types.state_input: self.StateInputEvoMatStruct(),
            self.matrix_types.output     : self.OutputEvoMatStruct(),
            self.matrix_types.constraint : self.ConstraintEvoMatStruct()
        })

    def __init__(self, controller=None, N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet):
        super(MldEvoMatrices, self).__init__(controller, N_p=N_p, N_tilde=N_tilde,
                                             mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)
        self.update(reset=True)

    def __repr__(self):
        base_repr = super(MldEvoMatrices, self).__repr__()
        mld_evo_mat_str = ("\n"
                         "              x_N_tilde = Phi_x_N_tilde @ x(0) + Gamma_v_N_tilde @ v_N_tilde\n"
                         "                          + Gamma_omega_N_tilde @ omega_N_tilde\n"
                         "                          + Gamma_5_N_tilde\n"
                         "              y_N_tilde = L_x_N_tilde @ x(0) + L_v_N_tilde @ v_N_tilde\n"
                         "                          + L_omega_N_tilde @ omega_N_tilde\n"
                         "                          + L_5_N_tilde\n"
                         "H_v_N_tilde @ v_N_tilde <= H_x_N_tilde @ x(0) + H_omega_N_tilde @ omega_N_tilde\n"
                         "                          + H_5_N_tilde\n"
                         "\n"
                         "with:\n")
        mld_evo_mat_repr = base_repr.replace('{', '{\n' + mld_evo_mat_str, 1)
        return mld_evo_mat_repr

    @increments_version_decor
    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def update(self, N_p=None, N_tilde=None,
               mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
               A_pow_tilde=None, sparse=None, mat_ops=None, reset=False):

        if reset or self.has_updated_version(sub_object_names=('mld_numeric_k', 'mld_numeric_tilde')):
            self._base_dict_update(
                self.gen_mld_evo_matrices(_disable_process_args=True, N_p=N_p, N_tilde=N_tilde,
                                          mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde,
                                          mld_info_k=mld_info_k,
                                          A_pow_tilde=A_pow_tilde,
                                          sparse=sparse, mat_ops=mat_ops, reset=True))
            self.update_stored_version()
            self._update_set_with(N_p, N_tilde)

    @process_method_args_decor(process_base_args)
    def get_evo_matrices_N_tilde(self, N_tilde=None, mld_info_k: MldInfo = None):
        if N_tilde == self.N_tilde:
            return self
        elif N_tilde > self.N_tilde:
            raise ValueError(f"N_tilde:{N_tilde} cannot be greater than self.N_tilde:{self.N_tilde}")

        evo_matrices_N_tilde = self._constructor_from_self(copy_items=True)

        for mat_type, dim_name in zip(self.matrix_types, MldInfo._sys_dim_names):
            row_partition_size = mld_info_k[dim_name]
            for evo_mat_N_tilde_name, evo_mat_N_tilde in self[mat_type].items():
                if evo_mat_N_tilde_name.endswith('N_tilde'):
                    evo_matrices_N_tilde[mat_type][evo_mat_N_tilde_name] = (
                        self[mat_type][evo_mat_N_tilde_name][:N_tilde * row_partition_size, :])

        return evo_matrices_N_tilde

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def gen_mld_evo_matrices(self, N_p=None, N_tilde=None,
                             mld_numeric_k=None, mld_numeric_tilde=None, mld_info_k=None,
                             A_pow_tilde=None, sparse=None, mat_ops=None, reset=False):

        gen_kwargs = dict(_disable_process_args=True, N_tilde=N_tilde,
                          mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde, mld_info_k=mld_info_k,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)
        mpc_evo_struct = MldEvoMatricesStruct({self.matrix_types.state_input: self.StateInputEvoMatStruct(),
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
            self.gen_constraint_evo_matrices(N_p=N_p,
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

        self._update_with_N_p_slices(state_input_evo_struct, mld_info_k.n_states, N_p)

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

        self._update_with_N_p_slices(output_evo_struct, mld_info_k.n_outputs, N_p)

        return output_evo_struct

    @process_method_args_decor(process_base_args, process_mat_op_args, process_A_pow_tilde_arg)
    def gen_constraint_evo_matrices(self, N_p=None, N_tilde=None,
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

        self._update_with_N_p_slices(cons_evo_struct, mld_info_k.n_constraints, N_p)

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

        non_empty_mats = [mat_name for mat_name in mat_names if mat_name not in mld_numeric_k._all_empty_mats]
        non_zero_mats = [mat_name for mat_name in mat_names if mat_name not in mld_numeric_k._all_zero_mats]

        mat_hstack_k = mat_ops.vmatrix(mat_ops.package.hstack(
            [mat_ops.hmatrix(mld_numeric_k[mat_name]) for mat_name in mat_names]))

        if non_zero_mats:  # constant non-zero matrices exist
            if mld_numeric_tilde:
                mat_hstack_tilde = [
                    mat_ops.vmatrix(mat_ops.package.hstack(
                        [mat_ops.hmatrix(mld_numeric_tilde[k][mat_name]) for mat_name in non_empty_mats])
                    ) for k in range(N_tilde)]

            else:
                mat_hstack_tilde = [mat_hstack_k] * N_tilde

            return mat_ops.block_diag(mat_hstack_tilde)
        else:
            return mat_ops.zeros(tuple(np.array(mat_hstack_k.shape) * N_tilde))
