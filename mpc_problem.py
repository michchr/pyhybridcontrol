import scipy.sparse as scs
import scipy.linalg as scl
import numpy as np

from structdict import StructDict

from model_generators import DewhModelGenerator, GridModelGenerator
from dewh_repository import DewhRepository
from micro_grid_model import MicroGridModel


class MpcProblem():

    def __init__(self, micro_grid_model: MicroGridModel):
        self._micro_grid_model = micro_grid_model
        self._mg_device_state_evo_struct = None
        self._mg_device_cons_evo_struct = None
        self._mg_grid_cons_evo_stuct = None
        self._mg_mpc_cons_struct = None

    @property
    def micro_grid_model(self):
        return self._micro_grid_model

    @micro_grid_model.setter
    def micro_grid_model(self, micro_grid_model):
        self._micro_grid_model = micro_grid_model

    def gen_device_state_evolution_matrices(self, N_p, micro_grid_model=None):
        if micro_grid_model is None:
            micro_grid_model = self.micro_grid_model

        sys_m = micro_grid_model.device_mld_mat_struct

        mg_device_state_evo_struct = StructDict()

        B_s123 = scs.hstack([sys_m.B_s1, sys_m.B_s2, sys_m.B_s3])
        A_s_pow = [(sys_m.A_s ** i).tocsc() for i in range(N_p + 1)]

        # State evolution matrices
        # x_s_tilde = Phi_V_s*V_s_tilde + Phi_W_s*W_s_tilde + Phi_b_s5*b_s5_tilde

        # TEMP CODE##

        def mat_evo_gen(N_p, A, A_pow=None, B=None, with_zero=True):
            if A_pow is None:
                A_pow = [(A ** i).tocsc() for i in range(N_p + 1)]

            if B is None:
                B = A
                np_array_toep = scl.toeplitz(
                    [scs.csc_matrix(B.shape)] + [A_pow[i] for i in range(N_p)],
                    [scs.csc_matrix(B.shape)] * (N_p))
            else:
                np_array_toep = scl.toeplitz(
                    [scs.csc_matrix(B.shape)] + [A_pow[i] * B for i in range(N_p)],
                    [scs.csc_matrix(B.shape)] * (N_p))

            n_row = B.shape[0] * (N_p + 1) if with_zero else B.shape[0] * (N_p)
            n_col = B.shape[1] * N_p

            h_stacked = np.empty((np_array_toep.shape[0],1), dtype=object)
            for ind, row in enumerate(np_array_toep):
                h_stacked[ind] = scs.hstack(row).tocsr()


            return scs.bmat(h_stacked)

        # Phi_V_s_sparse = scs.bmat(scl.toeplitz(
        #     [scs.csc_matrix(B_s123.shape)] + [A_s_pow[i] * B_s123 for i in range(N_p)],
        #     [scs.csc_matrix(B_s123.shape)] * (N_p)))
        #
        # Phi_V_s_sparse_2 = mat_evo_gen(N_p, sys_m.A_s, A_s_pow, B_s123)
        #
        # pprint.pprint(np.allclose(Phi_V_s_sparse.A,Phi_V_s_sparse_2.A))
        # pprint.pprint(Phi_V_s_sparse_2.A)

        ##END TEMP CODE##

        mg_device_state_evo_struct.Phi_x_s = scs.vstack(A_s_pow)
        mg_device_state_evo_struct.Phi_V_s = scs.bmat(
            scl.toeplitz([scs.coo_matrix(B_s123.shape)] + [A_s_pow[i] * B_s123 for i in range(N_p)],
                         [scs.coo_matrix(B_s123.shape)] * (N_p)))
        mg_device_state_evo_struct.Phi_W_s = scs.bmat(
            scl.toeplitz([scs.coo_matrix(sys_m.B_s4.shape)] + [A_s_pow[i] * sys_m.B_s4 for i in range(N_p)],
                         [scs.coo_matrix(sys_m.B_s4.shape)] * (N_p)))
        mg_device_state_evo_struct.Phi_b_s5 = scs.bmat(
            scl.toeplitz([scs.coo_matrix(sys_m.A_s.shape)] + [A_s_pow[i] for i in range(N_p)],
                         [scs.coo_matrix(sys_m.A_s.shape)] * (N_p)))

        # Phi_V_s = mat_evo_gen(N_p, sys_m.A_s, A_s_pow, B_s123)
        # Phi_W_s = mat_evo_gen(N_p, sys_m.A_s, A_s_pow, sys_m.B_s4)
        # Phi_b_s5 = mat_evo_gen(N_p, sys_m.A_s, A_s_pow)

        mg_device_state_evo_struct.b_s5_tilde = np.vstack([sys_m.b_s5] * (N_p))

        self._mg_device_state_evo_struct = mg_device_state_evo_struct

        return mg_device_state_evo_struct

    def gen_device_cons_evolution_matrices(self, N_p, micro_grid_model=None, mg_device_state_evo_struct=None):
        if micro_grid_model is None:
            micro_grid_model = self.micro_grid_model

        if mg_device_state_evo_struct is None:
            mg_device_state_evo_struct = self._mg_device_state_evo_struct

        dev_mld = micro_grid_model.device_mld_mat_struct
        dev_evo = mg_device_state_evo_struct

        ## System Constraint Matrices
        E_s1_tilde = scs.block_diag([dev_mld.E_s1] * (N_p + 1))
        E_s234_tilde = scs.block_diag([scs.hstack([dev_mld.E_s2, dev_mld.E_s3, dev_mld.E_s4])] * (N_p + 1))
        E_s5_tilde = scs.block_diag([dev_mld.E_s5] * (N_p + 1))
        d_s_tilde = np.vstack([dev_mld.d_s] * (N_p + 1))

        ## Device Constraint Matrices i.e. H_s_V_s*V_s <= H_s_5_s - H_s_W_s*W_s - H_s_x_s*x_s
        mg_device_cons_evo_struct = StructDict()

        try:  # E_s234_tilde may be a null matrix
            mg_device_cons_evo_struct.H_s_V_s = (E_s1_tilde + E_s234_tilde) * dev_evo.Phi_V_s  #
        except ValueError:
            mg_device_cons_evo_struct.H_s_V_s = (E_s1_tilde) * dev_evo.Phi_V_s

        try:  # E_s5_tilde may be a null matrix
            mg_device_cons_evo_struct.H_s_W_s = E_s1_tilde * dev_evo.Phi_W_s + E_s5_tilde
        except ValueError:
            mg_device_cons_evo_struct.H_s_W_s = E_s1_tilde * dev_evo.Phi_W_s

        mg_device_cons_evo_struct.H_s_5_s = d_s_tilde - E_s1_tilde * dev_evo.Phi_b_s5 * dev_evo.b_s5_tilde
        mg_device_cons_evo_struct.H_s_x_s = E_s1_tilde * dev_evo.Phi_x_s

        self._mg_device_cons_evo_struct = mg_device_cons_evo_struct

        return mg_device_cons_evo_struct, mg_device_state_evo_struct

    def gen_grid_cons_evolution_matrices(self, N_p, micro_grid_model=None):
        if micro_grid_model is None:
            micro_grid_model = self.micro_grid_model

        sum_load_vec = micro_grid_model.get_grid_summation_vector()

        Gam_V_s = scs.block_diag([sum_load_vec] * N_p)
        Gam_W_p = scs.block_diag([np.array([1, -1])] * N_p)

        grid_mld = micro_grid_model.grid_mld_mat_struct

        E_p2_tilde = scs.block_diag([grid_mld.E_p2] * N_p)

        E_p34 = np.hstack([grid_mld.E_p3, grid_mld.E_p4])
        E_p34_tilde = scs.block_diag([E_p34] * N_p)

        d_p_tilde = np.vstack([grid_mld.d_p] * N_p)

        ## Grid Constraint Matrices i.e. H_p_V_s*V_s + H_p_V_p*V_p <= H_p_5_p - H_p_W_p*W_p
        mg_grid_cons_evo_struct = StructDict()

        mg_grid_cons_evo_struct.H_p_V_s = E_p2_tilde * Gam_V_s
        mg_grid_cons_evo_struct.H_p_V_p = E_p34_tilde
        mg_grid_cons_evo_struct.H_p_W_p = E_p2_tilde * Gam_W_p
        mg_grid_cons_evo_struct.H_p_5_p = d_p_tilde

        self._mg_device_cons_evo_struct = mg_grid_cons_evo_struct

        return mg_grid_cons_evo_struct

    def gen_mpc_cons_matrices(self, N_p, micro_grid_model=None):
        if micro_grid_model is None:
            micro_grid_model = self.micro_grid_model

        mg_device_state_evo_struct = self.gen_device_state_evolution_matrices(N_p, micro_grid_model)
        dev_con_evo, _ = mg_device_cons_evo_struct, _ = self.gen_device_cons_evolution_matrices(N_p, micro_grid_model,
                                                                                                mg_device_state_evo_struct)
        grid_con_evo = mg_grid_cons_evo_struct = self.gen_grid_cons_evolution_matrices(N_p, micro_grid_model)

        ## OVERALL CONSTRAINT MATRICES AND VECTORS A*V <= b ==> F1*V <= F2 + F3w*W + F4x*x

        mg_mpc_cons_struct = StructDict()

        mg_mpc_cons_struct.F1 = scs.bmat(
            np.array([[dev_con_evo.H_s_V_s, None], [grid_con_evo.H_p_V_s, grid_con_evo.H_p_V_p]]))

        mg_mpc_cons_struct.F2 = scs.bmat(np.array([[dev_con_evo.H_s_5_s], [grid_con_evo.H_p_5_p]]))

        mg_mpc_cons_struct.F3w = scs.block_diag([-dev_con_evo.H_s_W_s, -grid_con_evo.H_p_W_p])

        mg_mpc_cons_struct.F4x = scs.vstack(
            [-dev_con_evo.H_s_x_s, scs.coo_matrix((grid_con_evo.H_p_5_p.shape[0], dev_con_evo.H_s_x_s.shape[1]))])

        self._mg_mpc_cons_struct = mg_mpc_cons_struct

        return mg_mpc_cons_struct


################################################################################
################################    MAIN     ###################################
################################################################################

if __name__ == '__main__':
    import timeit
    import pprint
    from parameters import dewh_p, grid_p


    def main():
        N_h = 10
        N_p = 96

        dewh_repo = DewhRepository(DewhModelGenerator)
        dewh_repo.default_param_struct = dewh_p

        for i in range(N_h):
            dewh_repo.add_device_by_default_data(i)

        mg_model = MicroGridModel()
        mg_model.grid_param_struct = grid_p

        mg_model.add_device_repository(dewh_repo)
        mg_model.gen_concat_device_system_mld()

        mg_model.gen_power_balance_constraint_mld()

        mpc_prob = MpcProblem(mg_model)

        # mpc_prob.gen_device_state_evolution_matrices(N_p)
        # mpc_prob.gen_device_cons_evolution_matrices(N_p)
        # mpc_prob.gen_grid_cons_evolution_matrices(N_p)

        mpc_prob.gen_mpc_cons_matrices(N_p)
        # pprint.pprint(mpc_prob.gen_mpc_cons_matrices(N_p))


    def func():
        def closure():
            main()
            return 1

        return closure


    t1 = timeit.timeit(func(), number=1)
    print(t1)
