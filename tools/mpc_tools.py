import itertools
import weakref

import cvxpy as cvx
import numpy as np

from utils.decorator_utils import ParNotSet, process_method_args_decor
from utils.matrix_utils import get_mat_ops, block_toeplitz
from utils.structdict import StructDict

from models.mld_model import MldModel, MldInfo


class MpcBase:
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

    def _process_base_args(self, f_kwargs=None, *,
                           N_p=ParNotSet, include_term_cons=ParNotSet,
                           mld_numeric=ParNotSet, mld_numeric_tilde=ParNotSet):

        if N_p is None:
            f_kwargs['N_p'] = self.N_p

        if include_term_cons is None:
            f_kwargs['include_term_cons'] = self.include_term_cons

        if mld_numeric_tilde is None:
            f_kwargs['mld_numeric_tilde'] = self.mld_numeric_tilde

        if mld_numeric is None:
            f_kwargs['mld_numeric'] = self.mld_numeric

        return f_kwargs

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


class MpcOptVariables(MpcBase):
    def __init__(self, agent=None, N_p=None, include_term_cons=None, mld_numeric=None, mld_numeric_tilde=None):
        super(MpcOptVariables, self).__init__(agent=agent, N_p=N_p, include_term_cons=include_term_cons,
                                              mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde)

    @process_method_args_decor('_process_base_args')
    def gen_optimization_vars(self, N_p=None, include_term_cons=None, mld_numeric=None, mld_numeric_tilde=None):
        N_cons = N_p + 1 if include_term_cons else N_p

        mld_info: MldInfo = mld_numeric_tilde[0].mld_info if mld_numeric_tilde else mld_numeric.mld_info

        var_names = mld_info._var_names_controllable
        V_opt_var_names = ['u', 'delta', 'z']

        #extract variable types from mld mld_info's
        if mld_numeric_tilde:
            var_types_tilde_mats = {
                var_name: (
                    np.hstack(
                        [mld_numeric_tilde[k].mld_info.get_var_type(var_name) for k in range(N_cons)]
                    ) if mld_info.get_var_dim(var_name) else np.empty((0, mld_info.get_var_dim(var_name)), dtype=np.str)
                ) for var_name in var_names
            }
        else:
            var_types_tilde_mats = {
                var_name: (
                    np.repeat(mld_numeric.mld_info.get_var_type(var_name), N_cons, axis=1))
                for var_name in var_names
            }

        def to_bin_index (type_mat): return(
            list(map(tuple, np.argwhere(type_mat == 'b').tolist())))

        #generate individual variable tilde mats
        opt_var_tilde_mats = {
            var_name: (
                cvx.Variable(var_type_mat.shape, boolean=to_bin_index(var_type_mat)) if var_type_mat.size
                else np.empty((0, N_cons))
            ) for var_name, var_type_mat in var_types_tilde_mats.items()
        }

        #add combined input variable tilde mat
        opt_var_tilde_mats['v'] = cvx.vstack(
            [opt_var_tilde_mats[var_name] for var_name in V_opt_var_names if opt_var_tilde_mats[var_name]])

        opt_var_struct = StructDict()

        def app_tilde(var_name, postfix): return(
            "".join([var_name.capitalize(), '_tilde_', postfix]))

        #add named tilde_mat_N_cons to output
        opt_var_struct.update({
            app_tilde(var_name, 'mat_N_cons'): (opt_var_tilde_mat)
            for var_name, opt_var_tilde_mat in opt_var_tilde_mats.items()}
        )

        #add named tilde_N_cons to output
        opt_var_struct.update({
            app_tilde(var_name, 'N_cons'): (
                cvx.reshape(var_mat_tilde, (var_mat_tilde.size, 1)) if var_mat_tilde.size else np.empty((0, 1))
            ) for var_name, var_mat_tilde in opt_var_tilde_mats.items()
        })

        #add named tilde_N_p to output
        opt_var_struct.update({
            app_tilde(var_name, 'N_p'): (
                cvx.reshape(var_mat_tilde[:, :N_p],
                            (var_mat_tilde.shape[0] * N_p, 1)) if var_mat_tilde.size else np.empty((0, 1))
            ) for var_name, var_mat_tilde in opt_var_tilde_mats.items()
        })

        return opt_var_struct


class MpcEvoGenerator(MpcBase):
    def __init__(self, agent=None, N_p=None, include_term_cons=None, mld_numeric=None, mld_numeric_tilde=None):
        super(MpcEvoGenerator, self).__init__(agent=agent, N_p=N_p, include_term_cons=include_term_cons,
                                              mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde)

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
                self._gen_A_pow_tilde(_disable_process_args=True, N_p=f_kwargs['N_p'],
                                      mld_numeric=f_kwargs['mld_numeric'],
                                      mld_numeric_tilde=f_kwargs['mld_numeric_tilde'], sparse=f_kwargs['sparse'],
                                      mat_ops=f_kwargs['mat_ops']))

        return f_kwargs

    @process_method_args_decor('_process_base_args', '_process_mat_op_args', '_process_A_pow_tilde_arg')
    def gen_mpc_evolution_matrices(self, N_p=None, include_term_cons=None,
                                   mld_numeric=None, mld_numeric_tilde=None,
                                   A_pow_tilde=None, sparse=None, mat_ops=None):

        gen_kwargs = dict(_disable_process_args=True, N_p=N_p, include_term_cons=include_term_cons,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        state_input_evolution_struct = self.gen_state_input_evolution_matrices(**gen_kwargs)

        output_evolution_struct = self.gen_output_evolution_matrices(**gen_kwargs)

        gen_kwargs['state_input_evolution_struct'] = state_input_evolution_struct
        gen_kwargs['output_evolution_struct'] = output_evolution_struct

        cons_evolution_struct = self.gen_cons_evolution_matrices(**gen_kwargs)

        mpc_evolution_struct = StructDict()
        mpc_evolution_struct.update(state_input_evolution_struct)
        mpc_evolution_struct.update(output_evolution_struct)
        mpc_evolution_struct.update(cons_evolution_struct)

        return mpc_evolution_struct

    @process_method_args_decor('_process_base_args', '_process_mat_op_args', '_process_A_pow_tilde_arg')
    def gen_state_input_evolution_matrices(self, N_p=None, include_term_cons=None,
                                           mld_numeric=None, mld_numeric_tilde=None,
                                           A_pow_tilde=None, sparse=None, mat_ops=None):

        # X_tilde = Gamma_V @ V + Gamma_W @ W + Gamma_b5

        gen_kwargs = dict(_disable_process_args=True, N_p=N_p, include_term_cons=include_term_cons,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        state_input_evolution_struct = StructDict()
        state_input_evolution_struct['Phi_x'] = Phi_x = self._gen_phi_x(**gen_kwargs)
        state_input_evolution_struct['Gamma_V'] = Gamma_V = self._gen_gamma_V(**gen_kwargs)
        state_input_evolution_struct['Gamma_W'] = Gamma_W = self._gen_gamma_W(**gen_kwargs)
        state_input_evolution_struct['Gamma_b5'] = Gamma_b5 = self._gen_gamma_b5(**gen_kwargs)

        mld_info = mld_numeric_tilde[0].mldinfo if mld_numeric_tilde else mld_numeric.mld_info
        nx, nu, ndelta, nz, nomega = mld_info.get_sub_list(['nx', 'nu', 'ndelta', 'nz', 'nomega'])

        N_cons = N_p + 1 if include_term_cons else N_p
        state_input_evolution_struct['Phi_x_N_cons'] = Phi_x[:N_cons * nx, :]
        state_input_evolution_struct['Gamma_V_N_cons'] = Gamma_V[:N_cons * nx, :]
        state_input_evolution_struct['Gamma_W_N_cons'] = Gamma_W[:N_cons * nx, :]
        state_input_evolution_struct['Gamma_b5_N_cons'] = Gamma_b5[:N_cons * nx, :]

        return state_input_evolution_struct

    @process_method_args_decor('_process_base_args', '_process_mat_op_args', '_process_A_pow_tilde_arg')
    def gen_output_evolution_matrices(self, N_p=None, include_term_cons=None,
                                      mld_numeric=None, mld_numeric_tilde=None,
                                      state_input_evolution_struct=None,
                                      A_pow_tilde=None, sparse=None, mat_ops=None):

        # Y_tilde = L_x @ x_0 + L_V @ V_tilde + L_W @ W_tilde + L_5

        gen_kwargs = dict(_disable_process_args=True, N_p=N_p, include_term_cons=include_term_cons,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        if state_input_evolution_struct is None:
            state_input_evolution_struct = self.gen_state_input_evolution_matrices(**gen_kwargs)

        Phi_x_N_cons, Gamma_V_N_cons, Gamma_W_N_cons, Gamma_b5_N_cons = (
            state_input_evolution_struct.get_sub_list(
                ['Phi_x_N_cons', 'Gamma_V_N_cons', 'Gamma_W_N_cons', 'Gamma_b5_N_cons']
            ))

        gen_kwargs.pop('A_pow_tilde')  # A_pow_tilde not required for output evo gen matrices

        output_evo_struct = StructDict()

        C_tilde = self._gen_C_tilde_diag(**gen_kwargs)
        D_123_tilde = self._gen_D123_tilde_diag(**gen_kwargs)
        D4_tilde = self._gen_D4_tilde_diag(**gen_kwargs)
        d5_tilde = self._gen_d5_tilde(**gen_kwargs)

        output_evo_struct['L_x_N_cons'] = C_tilde @ Phi_x_N_cons
        output_evo_struct['L_V_N_cons'] = C_tilde @ Gamma_V_N_cons + D_123_tilde
        output_evo_struct['L_W_N_cons'] = C_tilde @ Gamma_W_N_cons + D4_tilde
        output_evo_struct['L_5_N_cons'] = C_tilde @ Gamma_b5_N_cons + d5_tilde

        return output_evo_struct

    @process_method_args_decor('_process_base_args', '_process_mat_op_args', '_process_A_pow_tilde_arg')
    def gen_cons_evolution_matrices(self, N_p=None, include_term_cons=None,
                                    mld_numeric=None, mld_numeric_tilde=None,
                                    state_input_evolution_struct=None,
                                    output_evolution_struct=None,
                                    A_pow_tilde=None, sparse=None, mat_ops=None):

        # H_V @ V_tilde <= H_x @ x_0 + H_W @ W_tilde + H_5

        gen_kwargs = dict(_disable_process_args=True, N_p=N_p, include_term_cons=include_term_cons,
                          mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                          A_pow_tilde=A_pow_tilde,
                          sparse=sparse, mat_ops=mat_ops)

        if state_input_evolution_struct is None:
            state_input_evolution_struct = self.gen_state_input_evolution_matrices(**gen_kwargs)

        Phi_x_N_cons, Gamma_V_N_cons, Gamma_W_N_cons, Gamma_b5_N_cons = (
            state_input_evolution_struct.get_sub_list(
                ['Phi_x_N_cons', 'Gamma_V_N_cons', 'Gamma_W_N_cons', 'Gamma_b5_N_cons']
            ))

        if output_evolution_struct is None:
            output_evolution_struct = self.gen_output_evolution_matrices(
                state_input_evolution_struct=state_input_evolution_struct, **gen_kwargs)

        L_x_N_cons, L_V_N_cons, L_W_N_cons, L_5_N_cons = (
            output_evolution_struct.get_sub_list(
                ['L_x_N_cons', 'L_V_N_cons', 'L_W_N_cons', 'L_5_N_cons']
            ))

        gen_kwargs.pop('A_pow_tilde')  # A_pow_tilde not required for constraint evo gen matrices

        cons_evo_struct = StructDict()

        E_tilde = self._gen_E_tilde_diag(**gen_kwargs)
        F_123_tilde = self._gen_F123_tilde_diag(**gen_kwargs)
        F4_tilde = self._gen_F4_tilde_diag(**gen_kwargs)
        f5_tilde = self._gen_f5_tilde(**gen_kwargs)
        G_tilde = self._gen_G_tilde_diag(**gen_kwargs)
        Psi_tilde = self._gen_Psi_tilde_diag(**gen_kwargs)

        cons_evo_struct['H_x'] = - (E_tilde @ Phi_x_N_cons + G_tilde @ L_x_N_cons)
        cons_evo_struct['H_V'] = E_tilde @ Gamma_V_N_cons + F_123_tilde + G_tilde @ L_V_N_cons
        cons_evo_struct['H_W'] = -(E_tilde @ Gamma_W_N_cons + F4_tilde + G_tilde @ L_W_N_cons)
        cons_evo_struct['H_5'] = f5_tilde - (E_tilde @ Gamma_b5_N_cons + G_tilde @ L_5_N_cons)

        return cons_evo_struct

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_A_pow_tilde(self, N_p=None, mld_numeric=None, mld_numeric_tilde=None,
                         sparse=None, mat_ops=None):

        # A_pow_tilde = [(A_k)^0, (A_k+1)^1, (A_k+2)^2, ..., (A_k+N_p)^(N_p)]
        if mld_numeric_tilde:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric_tilde[0].A.shape))] + (
                [mat_ops.vmatrix(mld_numeric_tilde[k].A) for k in range(N_p)])
        else:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric.A.shape))] + [mat_ops.vmatrix(mld_numeric.A)] * (N_p)

        return tuple(itertools.accumulate(A_tilde, lambda x, y: mat_ops.vmatrix(x @ y)))

    @process_method_args_decor('_process_base_args', '_process_mat_op_args', )
    def _gen_phi_x(self, N_p=None, mld_numeric=None, mld_numeric_tilde=None,
                   A_pow_tilde=None, sparse=None, mat_ops=None, copy=None, **kwargs):

        # Phi_x = [(A_k)^0; (A_k+1)^1; (A_k+2)^2; ... ;(A_k+N_p)^(N_p)]
        Phi_x = mat_ops.package.vstack(A_pow_tilde)
        return Phi_x

    @process_method_args_decor('_process_base_args', '_process_mat_op_args', '_process_A_pow_tilde_arg')
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

    @process_method_args_decor('_process_base_args', '_process_mat_op_args', '_process_A_pow_tilde_arg')
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

    @process_method_args_decor('_process_base_args', '_process_mat_op_args', '_process_A_pow_tilde_arg')
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

    ### OUTPUT EVOLUTION MATRIX COMPONENTS ###

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_C_tilde_diag(self, N_p=None, include_term_cons=None,
                          mld_numeric=None, mld_numeric_tilde=None,
                          sparse=None, mat_ops=None):

        output_mat_names = ['C']

        C_tilde = self._gen_mat_tilde_diag(N_p=N_p, include_term_cons=include_term_cons,
                                           mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                           mat_names=output_mat_names,
                                           sparse=sparse, mat_ops=mat_ops)

        return C_tilde

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_D123_tilde_diag(self, N_p=None, include_term_cons=None,
                             mld_numeric=None, mld_numeric_tilde=None,
                             sparse=None, mat_ops=None):

        output_mat_names = ['D1', 'D2', 'D3']

        D_123_tilde = self._gen_mat_tilde_diag(N_p=N_p, include_term_cons=include_term_cons,
                                               mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                               mat_names=output_mat_names,
                                               sparse=sparse, mat_ops=mat_ops)

        return D_123_tilde

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_D4_tilde_diag(self, N_p=None, include_term_cons=None,
                           mld_numeric=None, mld_numeric_tilde=None,
                           sparse=None, mat_ops=None):

        output_mat_names = ['D4']

        D4_tilde = self._gen_mat_tilde_diag(N_p=N_p, include_term_cons=include_term_cons,
                                            mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                            mat_names=output_mat_names,
                                            sparse=sparse, mat_ops=mat_ops)

        return D4_tilde

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_d5_tilde(self, N_p=None, include_term_cons=None,
                      mld_numeric=None, mld_numeric_tilde=None,
                      sparse=None, mat_ops=None):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            d5_tilde = np.vstack([mld_numeric_tilde[k]['d5'] for k in range(N_cons)])
        else:
            d5_tilde = np.vstack([mld_numeric['d5']] * (N_cons))

        return d5_tilde

    ### CONSTRAINT EVOLUTION MATRIX COMPONENTS ###

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_E_tilde_diag(self, N_p=None, include_term_cons=None,
                          mld_numeric=None, mld_numeric_tilde=None,
                          sparse=None, mat_ops=None):

        cons_mat_names = ['E']

        E_tilde = self._gen_mat_tilde_diag(N_p=N_p, include_term_cons=include_term_cons,
                                           mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                           mat_names=cons_mat_names,
                                           sparse=sparse, mat_ops=mat_ops)

        return E_tilde

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_F123_tilde_diag(self, N_p=None, include_term_cons=None,
                             mld_numeric=None, mld_numeric_tilde=None,
                             sparse=None, mat_ops=None):

        cons_mat_names = ['F1', 'F2', 'F3']

        F_123_tilde = self._gen_mat_tilde_diag(N_p=N_p, include_term_cons=include_term_cons, mld_numeric=mld_numeric,
                                               mld_numeric_tilde=mld_numeric_tilde,
                                               mat_names=cons_mat_names,
                                               sparse=sparse, mat_ops=mat_ops)

        return F_123_tilde

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_F4_tilde_diag(self, N_p=None, include_term_cons=None,
                           mld_numeric=None, mld_numeric_tilde=None,
                           sparse=None, mat_ops=None):

        cons_mat_names = ['F4']

        F4_tilde = self._gen_mat_tilde_diag(N_p=N_p, include_term_cons=include_term_cons, mld_numeric=mld_numeric,
                                            mld_numeric_tilde=mld_numeric_tilde,
                                            mat_names=cons_mat_names,
                                            sparse=sparse, mat_ops=mat_ops)

        return F4_tilde

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_f5_tilde(self, N_p=None, include_term_cons=None,
                      mld_numeric=None, mld_numeric_tilde=None,
                      sparse=None, mat_ops=None):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            f5_tilde = np.vstack([mld_numeric_tilde[k]['f5'] for k in range(N_cons)])
        else:
            f5_tilde = np.vstack([mld_numeric['f5']] * (N_cons))

        return f5_tilde

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_G_tilde_diag(self, N_p=None, include_term_cons=None,
                          mld_numeric=None, mld_numeric_tilde=None,
                          sparse=None, mat_ops=None):

        cons_mat_names = ['G']

        G_tilde = self._gen_mat_tilde_diag(N_p=N_p, include_term_cons=include_term_cons, mld_numeric=mld_numeric,
                                           mld_numeric_tilde=mld_numeric_tilde,
                                           mat_names=cons_mat_names,
                                           sparse=sparse, mat_ops=mat_ops)

        return G_tilde

    @process_method_args_decor('_process_base_args', '_process_mat_op_args')
    def _gen_Psi_tilde_diag(self, N_p=None, include_term_cons=None,
                            mld_numeric=None, mld_numeric_tilde=None,
                            sparse=None, mat_ops=None):

        cons_mat_names = ['Psi']

        Psi_tilde = self._gen_mat_tilde_diag(N_p=N_p, include_term_cons=include_term_cons, mld_numeric=mld_numeric,
                                             mld_numeric_tilde=mld_numeric_tilde,
                                             mat_names=cons_mat_names,
                                             sparse=sparse, mat_ops=mat_ops)

        return Psi_tilde

    @staticmethod
    def _gen_input_evolution_mat(N_p=None, include_term_cons=None,
                                 mld_numeric=None, mld_numeric_tilde=None,
                                 input_mat_names=None, A_pow_tilde=None,
                                 sparse=None, mat_ops=None):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            B_hstack_tilde = [
                mat_ops.vmatrix(mat_ops.package.hstack(
                    [mat_ops.hmatrix(mld_numeric_tilde[k][input_mat_name]) for input_mat_name in input_mat_names])
                ) for k in range(N_p)]

            col_list = ([mat_ops.zeros(B_hstack_tilde[0].shape)] +
                        [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack_tilde[k]) for k in range(N_p)])
            row_list = [mat_ops.zeros(B_hstack_tilde[0].shape)] * (N_cons)
        else:
            B_hstack = mat_ops.vmatrix(mat_ops.package.hstack(
                [mat_ops.hmatrix(mld_numeric[input_mat_name]) for input_mat_name in input_mat_names]))

            col_list = ([mat_ops.zeros(B_hstack.shape)] +
                        [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack) for k in range(N_p)])
            row_list = [mat_ops.zeros(B_hstack.shape)] * (N_cons)

        return block_toeplitz(col_list, row_list, sparse=sparse)

    @staticmethod
    def _gen_mat_tilde_diag(N_p=None, include_term_cons=None,
                            mld_numeric=None, mld_numeric_tilde=None,
                            mat_names=None,
                            sparse=None, mat_ops=None):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            mat_hstack_tilde = [
                mat_ops.vmatrix(mat_ops.package.hstack(
                    [mat_ops.hmatrix(mld_numeric_tilde[k][cons_mat_name]) for cons_mat_name in mat_names])
                ) for k in range(N_cons)]

            mat_hstack_tilde_diag = mat_ops.block_diag(mat_hstack_tilde)
        else:
            mat_hstack_tilde = [mat_ops.vmatrix(mat_ops.package.hstack(
                [mat_ops.hmatrix(mld_numeric[cons_mat_name]) for cons_mat_name in mat_names]))] * N_cons

            mat_hstack_tilde_diag = mat_ops.block_diag(mat_hstack_tilde)

        return mat_hstack_tilde_diag
