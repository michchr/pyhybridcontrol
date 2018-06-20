from structdict import StructDict

import scipy.sparse as scs
import scipy.linalg as scl
import numpy as np

from model_generators import DewhModelGenerator, GridModelGenerator
from dewh_repository import DewhRepository
from micro_grid_model import MicroGridModel


class MpcProblem():

    def __init__(self, micro_grid_model: MicroGridModel):
        self._micro_grid_model = micro_grid_model
        self._mg_device_evo_mat_struct = None
        self._mg_mat_evo_device_struct = None

    @property
    def micro_grid_model(self):
        return self._micro_grid_model

    @micro_grid_model.setter
    def micro_grid_model(self, micro_grid_model):
        self._micro_grid_model = micro_grid_model

    def gen_device_state_evolution_matrices(self, N_p, micro_grid_model=None):
        if micro_grid_model == None:
            micro_grid_model = self._micro_grid_model

        sys_m = micro_grid_model.device_mld_mat_struct

        mg_device_evo_mat_struct = StructDict()

        B_s123 = scs.hstack([sys_m.B_s1, sys_m.B_s2, sys_m.B_s3])
        A_s_pow = [(sys_m.A_s ** i) for i in range(N_p + 1)]

        # State evolution matrices
        # x_s_tilde(k) = Phi_V_s*V_s_tilde + Phi_W_s*W_s_tilde + Phi_b_s5*b_s5_tilde

        mg_device_evo_mat_struct.Phi_x_s = scs.vstack(A_s_pow)
        mg_device_evo_mat_struct.Phi_V_s = scs.bmat(
            scl.toeplitz([scs.coo_matrix(B_s123.shape)] + [A_s_pow[i] * B_s123 for i in range(N_p)],
                         [scs.coo_matrix(B_s123.shape)] * (N_p)))
        mg_device_evo_mat_struct.Phi_W_s = scs.bmat(
            scl.toeplitz([scs.coo_matrix(sys_m.B_s4.shape)] + [A_s_pow[i] * sys_m.B_s4 for i in range(N_p)],
                         [scs.coo_matrix(sys_m.B_s4.shape)] * (N_p)))
        mg_device_evo_mat_struct.Phi_b_s5 = scs.bmat(
            scl.toeplitz([scs.coo_matrix(sys_m.A_s.shape)] + [A_s_pow[i] for i in range(N_p)],
                         [scs.coo_matrix(sys_m.A_s.shape)] * (N_p)))

        mg_device_evo_mat_struct.b_s5_tilde = np.vstack([sys_m.b_s5] * (N_p))

        self._mg_device_evo_mat_struct = mg_device_evo_mat_struct

        return mg_device_evo_mat_struct

    def gen_mpc_sys_state_cons(self, N_p, micro_grid_model=None, mg_device_evo_mat_struct=None):
        if micro_grid_model == None:
            micro_grid_model = self._micro_grid_model

        if mg_device_evo_mat_struct == None:
            mg_device_evo_mat_struct = self._mg_device_evo_mat_struct

        sys_m = micro_grid_model.device_mld_mat_struct
        dev_evo = mg_device_evo_mat_struct

        ## System Constraint Matrices
        E_s1_tilde = scs.block_diag([sys_m.E_s1] * (N_p + 1))
        E_s234_tilde = scs.block_diag([scs.hstack([sys_m.E_s2, sys_m.E_s3, sys_m.E_s4])] * (N_p + 1))
        E_s5_tilde = scs.block_diag([sys_m.E_s5] * (N_p + 1))
        d_s_tilde = np.vstack([sys_m.d_s] * (N_p + 1))

        ## Combined Constraint Matrices i.e. H_V_s*Vsv <= Hs5 - Hsw*Wsv - Hsx*xs
        mg_mat_evo_device_struct = StructDict()

        try:  # E_s234_tilde may be a null matrix
            mg_mat_evo_device_struct.H_V_s = (E_s1_tilde + E_s234_tilde) * dev_evo.Phi_V_s  #
        except ValueError:
            mg_mat_evo_device_struct.H_V_s = (E_s1_tilde) * dev_evo.Phi_V_s

        try:  # E_s5_tilde may be a null matrix
            mg_mat_evo_device_struct.H_W_s = E_s1_tilde * dev_evo.Phi_W_s + E_s5_tilde
        except ValueError:
            mg_mat_evo_device_struct.H_W_s = E_s1_tilde * dev_evo.Phi_W_s

        mg_mat_evo_device_struct.H_5_s = d_s_tilde - E_s1_tilde * dev_evo.Phi_b_s5 * dev_evo.b_s5_tilde
        mg_mat_evo_device_struct.H_x_s = E_s1_tilde * dev_evo.Phi_x_s

        self._mg_mat_evo_device_struct = mg_mat_evo_device_struct

        return mg_mat_evo_device_struct, mg_device_evo_mat_struct


if __name__ == '__main__':
    import timeit
    import pprint
    from parameters import dewh_p, grid_p


    def main():
        N_h = 2
        N_p = 5

        dewh_repo = DewhRepository(DewhModelGenerator)
        dewh_repo.default_dewh_param_struct = dewh_p

        for i in range(N_h):
            dewh_repo.add_dewh_by_default_data(i)

        mg_model = MicroGridModel()
        mg_model.grid_param_struct = grid_p

        mg_model.add_device_repository(dewh_repo)
        mg_model.gen_concat_device_system_mld()

        mg_model.gen_power_balance_constraint_mld()

        mpc_prob = MpcProblem(mg_model)

        mpc_prob.gen_device_state_evolution_matrices(N_p)

        pprint.pprint(mpc_prob._mg_device_evo_mat_struct)


    def func():
        def closure():
            main()
            return 1

        return closure


    t1 = timeit.timeit(func(), number=1)
    print(t1)
