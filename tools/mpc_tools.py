import itertools
import weakref

import cvxpy as cvx
import numpy as np

from utils.decorator_utils import process_method_args_decor, ParNotReq
from utils.matrix_utils import get_mat_ops, block_toeplitz
from utils.structdict import StructDict

class MpcEvoGenerator:

    def __init__(self, agent=None, N_p=None, include_term_cons=None, mld_numeric=None, mld_numeric_tilde=None):
        if agent is not None:
            self._agent = weakref.proxy(agent)
            self._N_p = None
            self._include_term_cons = None
            self._mld_numeric = None
            self._mld_numeric_tilde = None
        else:
            self._agent = None
            self._N_p = N_p or 0
            self._include_term_cons = include_term_cons or True
            self._mld_numeric = mld_numeric
            self._mld_numeric_tilde = mld_numeric_tilde

    @property
    def N_p(self):
        return (self._agent.N_p if self._agent else self._N_p)

    @property
    def include_term_cons(self):
        return (self._agent.include_term_cons if self._agent else self._include_term_cons)

    @property
    def mld_numeric(self):
        return (self._agent.mld_numeric if self._agent else self._mld_numeric)

    @property
    def mld_numeric_tilde(self):
        return (self._agent.mld_numeric_tilde if self._agent else self._mld_numeric_tilde)

    def _process_base_args(self, f_kwargs=None, *,
                           N_p=ParNotReq, include_term_cons=ParNotReq,
                           mld_numeric=ParNotReq, mld_numeric_tilde=ParNotReq,
                           sparse=ParNotReq, mat_ops=ParNotReq):

        if N_p is None:
            f_kwargs['N_p'] = self.N_p

        if include_term_cons is None:
            f_kwargs['include_term_cons'] = self.include_term_cons

        if sparse is None:
            f_kwargs['sparse'] = False

        if mat_ops is None:
            f_kwargs['mat_ops'] = get_mat_ops(sparse=f_kwargs['sparse'])

        if mld_numeric_tilde is None:
            f_kwargs['mld_numeric_tilde'] = self.mld_numeric_tilde

        if mld_numeric is None:
            f_kwargs['mld_numeric'] = self.mld_numeric

        return f_kwargs

    def _process_A_pow_tilde_arg(self, f_kwargs=None, *,
                                 A_pow_tilde=ParNotReq):

        if A_pow_tilde is None:
            f_kwargs['A_pow_tilde'] = (
                self._gen_A_pow_tilde(_disable_process_args=True, N_p=f_kwargs['N_p'],
                                      mld_numeric=f_kwargs['mld_numeric'],
                                      mld_numeric_tilde=f_kwargs['mld_numeric_tilde'], sparse=f_kwargs['sparse'],
                                      mat_ops=f_kwargs['mat_ops']))

        return f_kwargs

    @process_method_args_decor('_process_base_args', '_process_A_pow_tilde_arg')
    def gen_mpc_evolution_matrices(self, N_p=None, include_term_cons=None,
                                   mld_numeric=None, mld_numeric_tilde=None,
                                   A_pow_tilde=None, sparse=None, mat_ops=None):

        gen_kwargs = dict(_disable_process_args=True, N_p=N_p, include_term_cons=include_term_cons,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        state_input_evolution_struct = self.gen_state_input_evolution_matrices(**gen_kwargs)

        gen_kwargs['state_input_evolution_struct'] = state_input_evolution_struct

        cons_evolution_struct = self.gen_cons_evolution_matrices(**gen_kwargs)

        mpc_evolution_struct = StructDict(state_input_evolution_struct, **cons_evolution_struct)

        return mpc_evolution_struct

    @process_method_args_decor('_process_base_args', '_process_A_pow_tilde_arg')
    def gen_state_input_evolution_matrices(self, N_p=None, include_term_cons=None,
                                           mld_numeric=None, mld_numeric_tilde=None,
                                           A_pow_tilde=None, sparse=None, mat_ops=None):

        # X_tilde = Gamma_V @ V + Gamma_W @ W + Gamma_b5

        gen_kwargs = dict(_disable_process_args=True, N_p=N_p, include_term_cons=include_term_cons,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        state_input_evolution_struct = StructDict()
        state_input_evolution_struct['Phi_x'] = self._gen_phi_x(**gen_kwargs)
        state_input_evolution_struct['Gamma_V'] = self._gen_gamma_V(**gen_kwargs)
        state_input_evolution_struct['Gamma_W'] = self._gen_gamma_W(**gen_kwargs)
        state_input_evolution_struct['Gamma_b5'] = self._gen_gamma_b5(**gen_kwargs)

        return state_input_evolution_struct

    @process_method_args_decor('_process_base_args', '_process_A_pow_tilde_arg')
    def gen_cons_evolution_matrices(self, N_p=None, include_term_cons=None,
                                    mld_numeric=None, mld_numeric_tilde=None,
                                    state_input_evolution_struct=None,
                                    A_pow_tilde=None, sparse=None, mat_ops=None):

        # H_V @ V_tilde_N_con <= H_W @ W_tilde_N_con + H_5 + H_x @ x_0

        gen_kwargs = dict(_disable_process_args=True, N_p=N_p, include_term_cons=include_term_cons,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        if state_input_evolution_struct is None:
            state_input_evolution_struct = self.gen_state_input_evolution_matrices(**gen_kwargs)

        Phi_x = state_input_evolution_struct['Phi_x']
        Gamma_V = state_input_evolution_struct['Gamma_V']
        Gamma_W = state_input_evolution_struct['Gamma_W']
        Gamma_b5 = state_input_evolution_struct['Gamma_b5']

        if not include_term_cons:
            mld_info = mld_numeric_tilde[0].mldinfo if mld_numeric_tilde else mld_numeric.mld_info
            nx, nu, ndelta, nz, nomega = mld_info.get_sub_list(['nx', 'nu', 'ndelta', 'nz', 'nomega'])
            Phi_x = Phi_x[:N_p * nx, :]
            Gamma_V = Gamma_V[:N_p * (nu + ndelta + nz), :]
            Gamma_W = Gamma_W[:N_p * nomega, :]
            Gamma_b5 = Gamma_b5[:N_p * nx, :]

        gen_kwargs.pop('A_pow_tilde') #A_pow_tilde not required for constraint evo gen matrices

        cons_evo_struct = StructDict()

        E1_tilde = self._gen_E1_tilde_diag(**gen_kwargs)
        E_234_tilde = self._gen_E234_tilde_diag(**gen_kwargs)
        E5_tilde = self._gen_E5_tilde_diag(**gen_kwargs)
        g6_tilde = self._gen_g6_tilde_diag(**gen_kwargs)

        cons_evo_struct['H_x'] = - E1_tilde @ Phi_x
        cons_evo_struct['H_V'] = E1_tilde @ Gamma_V + E_234_tilde
        cons_evo_struct['H_W'] = -(E1_tilde @ Gamma_W + E5_tilde)
        cons_evo_struct['H_5'] = g6_tilde - E1_tilde @ Gamma_b5

        return cons_evo_struct

    @process_method_args_decor('_process_base_args')
    def _gen_optimization_vars(self, N_p=None, include_term_cons=None, mld_numeric=None, mld_numeric_tilde=None):
        N_cons = N_p + 1 if include_term_cons else N_p
        var_names = ['u', 'delta', 'z']
        var_type_map = {var_name: ("".join(["var_type_", var_name])) for var_name in var_names}

        if mld_numeric_tilde:
            V_types_tilde_mat = np.hstack([
                np.vstack(
                    [mld_numeric_tilde[k].mld_info[var_type_map[var_name]] for var_name in var_names]
                ) for k in range(N_cons)]
            )
            mld_info = mld_numeric_tilde[0].mld_info
        else:
            mld_info = mld_numeric.mld_info
            V_types_tilde_mat = np.hstack(
                [
                    np.vstack(
                        [mld_info[var_type_map[var_name]] for var_name in var_names]
                    )
                ] * N_cons
            )

        opt_var_struct = StructDict()

        bin_index = list(map(tuple, np.argwhere(V_types_tilde_mat == 'b').tolist()))
        opt_var_struct['V_tilde_mat_N_cons'] = V_tilde_mat_N_cons = cvx.Variable(V_types_tilde_mat.shape,
                                                                                 boolean=bin_index)

        nu = mld_info['nu']
        ndelta = mld_info['ndelta']
        nz = mld_info['nz']

        opt_var_struct['V_tilde_N_cons'] = cvx.reshape(V_tilde_mat_N_cons, (V_tilde_mat_N_cons.size, 1))
        opt_var_struct['U_tilde_N_cons'] = cvx.reshape(V_tilde_mat_N_cons[:nu, :], (N_cons * nu, 1))
        opt_var_struct['Delta_tilde_N_cons'] = cvx.reshape(V_tilde_mat_N_cons[nu:(nu + ndelta), :],
                                                           (N_cons * ndelta, 1))
        opt_var_struct['Z_tilde_N_cons'] = cvx.reshape(V_tilde_mat_N_cons[(nu + ndelta):, :], (N_cons * nz, 1))

        opt_var_struct['V_tilde_N_p'] = opt_var_struct['V_tilde_N_cons'][:N_p * (nu + ndelta + nz), :]
        opt_var_struct['U_tilde_N_p'] = opt_var_struct['U_tilde_N_cons'][:N_p * nu, :]
        opt_var_struct['Delta_tilde_N_p'] = opt_var_struct['Delta_tilde_N_cons'][:N_p * ndelta, :]
        opt_var_struct['Z_tilde_N_p'] = opt_var_struct['Z_tilde_N_cons'][:N_p * nz, :]
        return opt_var_struct

    @process_method_args_decor('_process_base_args')
    def _gen_A_pow_tilde(self, N_p=None, mld_numeric=None, mld_numeric_tilde=None,
                         sparse=None, mat_ops=None):

        # A_pow_tilde = [(A_k)^0, (A_k+1)^1, (A_k+2)^2, ..., (A_k+N_p)^(N_p)]
        if mld_numeric_tilde:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric_tilde[0].A.shape))] + (
                [mat_ops.vmatrix(mld_numeric_tilde[k].A) for k in range(N_p)])
        else:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric.A.shape))] + [mat_ops.vmatrix(mld_numeric.A)] * (N_p)

        return tuple(itertools.accumulate(A_tilde, lambda x, y: mat_ops.vmatrix(x @ y)))

    @process_method_args_decor('_process_base_args', '_process_A_pow_tilde_arg')
    def _gen_phi_x(self, N_p=None, mld_numeric=None, mld_numeric_tilde=None,
                   A_pow_tilde=None, sparse=None, mat_ops=None, copy=None, **kwargs):

        # Phi_x = [(A_k)^0; (A_k+1)^1; (A_k+2)^2; ... ;(A_k+N_p)^(N_p)]
        Phi_x = mat_ops.pack.vstack(A_pow_tilde)
        return Phi_x

    @process_method_args_decor('_process_base_args', '_process_A_pow_tilde_arg')
    def _gen_gamma_V(self, N_p=None, include_term_cons=None,
                     mld_numeric=None, mld_numeric_tilde=None,
                     A_pow_tilde=None,
                     sparse=None, mat_ops=None):
        # col = [[0s],(A_k)^0*[B1_k, B2_k, B3_k],..., (A_k+N_p-1)^(N_p-1)*[B1_k+N_p-1, B2_k+N_p-1, B3_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_V = toeplitz(col, row)

        input_mat_names = ['B1', 'B2', 'B3']
        Gamma_V = self._gen_input_evolution_mat(N_p=N_p, include_term_cons=include_term_cons,
                                                mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                                input_mat_names=input_mat_names,
                                                A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        return Gamma_V

    @process_method_args_decor('_process_base_args', '_process_A_pow_tilde_arg')
    def _gen_gamma_W(self, N_p=None, include_term_cons=None,
                     mld_numeric=None, mld_numeric_tilde=None,
                     A_pow_tilde=None,
                     sparse=None, mat_ops=None):

        # col = [[0s],(A_k)^0*[B4],..., (A_k+N_p-1)^(N_p-1)*[B4_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_W = toeplitz(col, row)

        input_mat_names = ['B4']
        Gamma_W = self._gen_input_evolution_mat(N_p=N_p, include_term_cons=include_term_cons,
                                                mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                                input_mat_names=input_mat_names,
                                                A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)
        return Gamma_W

    @process_method_args_decor('_process_base_args', '_process_A_pow_tilde_arg')
    def _gen_gamma_b5(self, N_p=None, include_term_cons=None,
                      mld_numeric=None, mld_numeric_tilde=None,
                      A_pow_tilde=None,
                      sparse=None, mat_ops=None):

        # col = [[0s],(A_k)^0*[b5],..., (A_k+N_p-1)^(N_p-1)*[b5_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_b5 = toeplitz(col, row)
        input_mat_names = ['b5']
        Gamma_b5_tilde = self._gen_input_evolution_mat(N_p=N_p, include_term_cons=include_term_cons,
                                                       mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                                       input_mat_names=input_mat_names,
                                                       A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        n, m = Gamma_b5_tilde.shape
        return Gamma_b5_tilde @ np.ones((m, 1))

    @process_method_args_decor('_process_base_args')
    def _gen_E1_tilde_diag(self, N_p=None, include_term_cons=None,
                           mld_numeric=None, mld_numeric_tilde=None,
                           sparse=None, mat_ops=None):

        cons_mat_names = ['E1']

        E1_tilde = self._gen_cons_tilde_diag(N_p=N_p, include_term_cons=include_term_cons,
                                             mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                             cons_mat_names=cons_mat_names,
                                             sparse=sparse, mat_ops=mat_ops)

        return E1_tilde

    @process_method_args_decor('_process_base_args')
    def _gen_E234_tilde_diag(self, N_p=None, include_term_cons=None,
                             mld_numeric=None, mld_numeric_tilde=None,
                             sparse=None, mat_ops=None):

        cons_mat_names = ['E2', 'E3', 'E4']

        E_234_tilde = self._gen_cons_tilde_diag(N_p=N_p, include_term_cons=include_term_cons, mld_numeric=mld_numeric,
                                                mld_numeric_tilde=mld_numeric_tilde,
                                                cons_mat_names=cons_mat_names,
                                                sparse=sparse, mat_ops=mat_ops)

        return E_234_tilde

    @process_method_args_decor('_process_base_args')
    def _gen_E5_tilde_diag(self, N_p=None, include_term_cons=None,
                           mld_numeric=None, mld_numeric_tilde=None,
                           sparse=None, mat_ops=None):

        cons_mat_names = ['E5']

        E5_tilde = self._gen_cons_tilde_diag(N_p=N_p, include_term_cons=include_term_cons, mld_numeric=mld_numeric,
                                             mld_numeric_tilde=mld_numeric_tilde,
                                             cons_mat_names=cons_mat_names,
                                             sparse=sparse, mat_ops=mat_ops)

        return E5_tilde

    @process_method_args_decor('_process_base_args')
    def _gen_g6_tilde_diag(self, N_p=None, include_term_cons=None,
                           mld_numeric=None, mld_numeric_tilde=None,
                           sparse=None, mat_ops=None):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            g6_tilde = np.vstack([mld_numeric_tilde[k].g6 for k in range(N_cons)])
        else:
            g6_tilde = np.vstack([mld_numeric.g6] * (N_cons))

        return g6_tilde

    @staticmethod
    def _gen_input_evolution_mat(N_p=None, include_term_cons=None,
                                 mld_numeric=None, mld_numeric_tilde=None,
                                 input_mat_names=None, A_pow_tilde=None,
                                 sparse=None, mat_ops=None):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            B_hstack_tilde = [
                mat_ops.vmatrix(mat_ops.pack.hstack(
                    [mat_ops.hmatrix(mld_numeric_tilde[k][input_mat_name]) for input_mat_name in input_mat_names])
                ) for k in range(N_p)]

            col_list = ([mat_ops.zeros(B_hstack_tilde[0].shape)] +
                        [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack_tilde[k]) for k in range(N_p)])
            row_list = [mat_ops.zeros(B_hstack_tilde[0].shape)] * (N_cons)
        else:
            B_hstack = mat_ops.vmatrix(mat_ops.pack.hstack(
                [mat_ops.hmatrix(mld_numeric[input_mat_name]) for input_mat_name in input_mat_names]))

            col_list = ([mat_ops.zeros(B_hstack.shape)] +
                        [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack) for k in range(N_p)])
            row_list = [mat_ops.zeros(B_hstack.shape)] * (N_cons)

        return block_toeplitz(col_list, row_list, sparse=sparse)

    @staticmethod
    def _gen_cons_tilde_diag(N_p=None, include_term_cons=None,
                             mld_numeric=None, mld_numeric_tilde=None,
                             cons_mat_names=None,
                             sparse=None, mat_ops=None):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            E_hstack_tilde = [
                mat_ops.vmatrix(mat_ops.pack.hstack(
                    [mat_ops.hmatrix(mld_numeric_tilde[k][cons_mat_name]) for cons_mat_name in cons_mat_names])
                ) for k in range(N_cons)]

            E_hstack_tilde_diag = mat_ops.block_diag(E_hstack_tilde)
        else:
            E_hstack_tilde = [mat_ops.vmatrix(mat_ops.pack.hstack(
                [mat_ops.hmatrix(mld_numeric[cons_mat_name]) for cons_mat_name in cons_mat_names]))] * N_cons

            E_hstack_tilde_diag = mat_ops.block_diag(E_hstack_tilde)

        return E_hstack_tilde_diag