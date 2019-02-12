import scipy.sparse as scs
import scipy.linalg as scl
import numpy as np

from structdict import StructDict


class MpcProblem():

    def __init__(self, micro_grid_model, N_p=None):
        self._mg_grid_cons_evo_stuct = None
        self._mg_tariff_generator = None
        self._micro_grid_model = micro_grid_model
        self._mg_device_state_evo_struct = None
        self._mg_device_cons_evo_struct = None

        self._mpc_cons_struct = None
        self._mpc_obj_struct = None
        self._decision_var_types = None

        self._N_p = N_p

    @property
    def N_p(self):
        return self._N_p

    @N_p.setter
    def N_p(self, N_p):
        self._N_p = N_p
        
    @property
    def mpc_obj_struct(self):
        return self._mpc_obj_struct
    
    @property
    def mpc_cons_struct(self):
        return self._mpc_cons_struct

    @property
    def decision_var_types(self):
        return self._decision_var_types

    @property
    def micro_grid_model(self):
        return self._micro_grid_model

    @micro_grid_model.setter
    def micro_grid_model(self, micro_grid_model):
        self._micro_grid_model = micro_grid_model

    @property
    def tariff_generator(self):
        return self._mg_tariff_generator

    @tariff_generator.setter
    def tariff_generator(self, tariff_generator):
        self._mg_tariff_generator = tariff_generator

    def gen_device_state_evolution_matrices(self, N_p=None, micro_grid_model=None):
        if N_p is None:
            N_p = self.N_p
        if micro_grid_model is None:
            micro_grid_model = self.micro_grid_model

        sys_m = micro_grid_model.device_mld_mat_struct

        mg_device_state_evo_struct = StructDict()

        B_s123 = scs.hstack([sys_m.B_s1, sys_m.B_s2, sys_m.B_s3])
        A_s_pow = [(sys_m.A_s ** i).tocsc() for i in range(N_p + 1)]

        # State evolution matrices
        # x_s_tilde = Phi_V_s*V_s_tilde + Phi_W_s*W_s_tilde + Phi_b_s5*b_s5_tilde

        mg_device_state_evo_struct.Phi_x_s = scs.vstack(A_s_pow)
        mg_device_state_evo_struct.Phi_V_s = mat_evo_gen(N_p, sys_m.A_s, A_s_pow, B_s123)
        mg_device_state_evo_struct.Phi_W_s = mat_evo_gen(N_p, sys_m.A_s, A_s_pow, sys_m.B_s4)
        mg_device_state_evo_struct.Phi_b_s5 = mat_evo_gen(N_p, sys_m.A_s, A_s_pow)

        mg_device_state_evo_struct.b_s5_tilde = np.vstack([sys_m.b_s5] * (N_p))

        self._mg_device_state_evo_struct = mg_device_state_evo_struct

        return mg_device_state_evo_struct

    def gen_device_cons_evolution_matrices(self, N_p=None, micro_grid_model=None, mg_device_state_evo_struct=None):
        if N_p is None:
            N_p = self.N_p

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

        ## Device Constraint Matrices i.e. H_s_V_s*V_s_tilde <= H_s_d_s - H_s_W_s*W_s_tilde - H_s_x_s*x_s
        mg_device_cons_evo_struct = StructDict()

        try:  # E_s234_tilde may be a null matrix
            mg_device_cons_evo_struct.H_s_V_s = (E_s1_tilde + E_s234_tilde) * dev_evo.Phi_V_s  #
        except ValueError:
            mg_device_cons_evo_struct.H_s_V_s = (E_s1_tilde) * dev_evo.Phi_V_s

        try:  # E_s5_tilde may be a null matrix
            mg_device_cons_evo_struct.H_s_W_s = E_s1_tilde * dev_evo.Phi_W_s + E_s5_tilde
        except ValueError:
            mg_device_cons_evo_struct.H_s_W_s = E_s1_tilde * dev_evo.Phi_W_s

        mg_device_cons_evo_struct.H_s_d_s = d_s_tilde - E_s1_tilde * dev_evo.Phi_b_s5 * dev_evo.b_s5_tilde
        mg_device_cons_evo_struct.H_s_x_s = E_s1_tilde * dev_evo.Phi_x_s

        self._mg_device_cons_evo_struct = mg_device_cons_evo_struct

        return mg_device_cons_evo_struct, mg_device_state_evo_struct

    def gen_grid_cons_evolution_matrices(self, N_p=None, micro_grid_model=None):
        if N_p is None:
            N_p = self.N_p

        if micro_grid_model is None:
            micro_grid_model = self.micro_grid_model

        sum_load_vec = micro_grid_model.get_grid_summation_vector()

        Gam_V_s = scs.block_diag([sum_load_vec.T] * N_p)
        Gam_W_p = scs.block_diag([np.array([1, -1])] * N_p)

        grid_mld = micro_grid_model.grid_mld_mat_struct

        E_p2_tilde = scs.block_diag([grid_mld.E_p2] * N_p)

        E_p34 = np.hstack([grid_mld.E_p3, grid_mld.E_p4])
        E_p34_tilde = scs.block_diag([E_p34] * N_p)

        d_p_tilde = np.vstack([grid_mld.d_p] * N_p)

        ## Grid Constraint Matrices i.e. H_p_V_s*V_s_tilde + H_p_V_p*V_p_tilde <= H_p_d_p_tilde - H_p_W_p*W_p_tilde
        mg_grid_cons_evo_struct = StructDict()

        mg_grid_cons_evo_struct.H_p_V_s = E_p2_tilde * Gam_V_s
        mg_grid_cons_evo_struct.H_p_V_p = E_p34_tilde
        mg_grid_cons_evo_struct.H_p_W_p = E_p2_tilde * Gam_W_p
        mg_grid_cons_evo_struct.H_p_d_p = d_p_tilde

        self._mg_device_cons_evo_struct = mg_grid_cons_evo_struct

        return mg_grid_cons_evo_struct

    def gen_mpc_objective(self, N_p=None, date_time_0=None, micro_grid_model=None, tariff_generator=None):
        if N_p is None:
            N_p = self.N_p
        if micro_grid_model is None:
            micro_grid_model = self.micro_grid_model
        if date_time_0 is None:
            date_time_0 = micro_grid_model.date_time_0
        if tariff_generator is None:
            tariff_generator = self.tariff_generator

        grid_p = micro_grid_model.grid_param_struct
        sum_load_vec = micro_grid_model.get_grid_summation_vector()

        C_imp_tilde = tariff_generator.get_price_vector(N_p, date_time_0, grid_p.ts)
        C_exp_tilde = np.zeros_like(C_imp_tilde) ## THIS WILL NEED TO CHANGE
        C_imp_sub_exp_tilde = C_imp_tilde # - C_exp_tilde

        S_V_s = (C_exp_tilde * sum_load_vec.T).reshape((-1, 1))
        S_V_p = (C_imp_sub_exp_tilde * np.array([[0, 1]])).reshape((-1, 1))

        S_W_s = np.zeros((micro_grid_model.device_mld_mat_struct.B_s4.shape[1] * N_p, 1))
        S_W_p = (C_exp_tilde * np.array([[1, -1]])).reshape((-1, 1))

        mpc_obj_struct = StructDict()

        mpc_obj_struct.S_V = np.vstack([S_V_s, S_V_p])
        mpc_obj_struct.S_W = np.vstack([S_W_s, S_W_p])

        self._mpc_obj_struct = mpc_obj_struct

        return mpc_obj_struct

    def gen_mpc_cons_matrices(self, N_p=None, micro_grid_model=None):
        if N_p is None:
            N_p = self.N_p
        if micro_grid_model is None:
            micro_grid_model = self.micro_grid_model

        mg_device_state_evo_struct = self.gen_device_state_evolution_matrices(N_p, micro_grid_model)
        dev_con_evo, _ = mg_device_cons_evo_struct, _ = self.gen_device_cons_evolution_matrices(
            N_p, micro_grid_model, mg_device_state_evo_struct)
        grid_con_evo = mg_grid_cons_evo_struct = self.gen_grid_cons_evolution_matrices(N_p, micro_grid_model)

        ## OVERALL CONSTRAINT MATRICES AND VECTORS A*V <= b ==> G_V*V <= G_d + G_W*W + G_x*x

        mpc_cons_struct = StructDict()


        mpc_cons_struct.G_V = scs.bmat(
            np.array([[dev_con_evo.H_s_V_s, None], [grid_con_evo.H_p_V_s, grid_con_evo.H_p_V_p]]))

        mpc_cons_struct.G_d = scs.bmat(np.array([[dev_con_evo.H_s_d_s], [grid_con_evo.H_p_d_p]]))

        mpc_cons_struct.G_W = scs.block_diag([-dev_con_evo.H_s_W_s, -grid_con_evo.H_p_W_p])

        mpc_cons_struct.G_x = scs.vstack(
            [-dev_con_evo.H_s_x_s, scs.coo_matrix((grid_con_evo.H_p_d_p.shape[0], dev_con_evo.H_s_x_s.shape[1]))])

        self._mpc_cons_struct = mpc_cons_struct

        return mpc_cons_struct

    def gen_mpc_decision_var_types(self, N_p=None, micro_grid_model=None):
        if N_p is None:
            N_p = self.N_p
        if micro_grid_model is None:
            micro_grid_model = self.micro_grid_model


        decision_var_types_devices = np.vstack([micro_grid_model.decision_var_types[0:-2]]*N_p)
        decision_var_types_grid = np.vstack([micro_grid_model.decision_var_types[-2::]]*N_p)

        decision_var_types = np.vstack([decision_var_types_devices, decision_var_types_grid])

        self._decision_var_types = decision_var_types

        return decision_var_types

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

    h_stacked = np.empty((np_array_toep.shape[0], 1), dtype=object)
    for ind, row in enumerate(np_array_toep):
        h_stacked[ind] = scs.hstack(row).tocsr()

    return scs.bmat(h_stacked)
