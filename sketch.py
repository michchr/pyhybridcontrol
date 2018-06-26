import numpy as np
import scipy as sc
import timeit
import functools

import scipy.linalg as scl
import scipy.sparse as scs
import scipy.sparse.linalg as scsl

# import pandas as pd
# import matplotlib.pylab as plt

import sympy as sp


def gen_dewh_symbolic_matrices():
    ts, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom = sp.symbols('ts, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom')
    T_h_min, T_h_max = sp.symbols('T_h_min, T_h_max')
    # p1, p2 = sp.symbols('p1, p2')
    p1 = U_h * A_h
    p2 = m_h * C_w

    # Define continuous system matrices
    A_h_D_c = -(D_h * C_w + p1) / p2
    B_h_D_c = sp.Matrix([[P_h_Nom, C_w * T_w, p1 * T_inf]]) * (p2 ** -1)

    # Compute discretized system matrices
    A_h_D = sp.exp(A_h_D_c * ts)
    B_h_D_3 = A_h_D_c ** (-1) * (sp.exp(A_h_D_c * ts) - 1) * B_h_D_c

    B_h1_D = B_h_D_3[0]
    B_h4_D = B_h_D_3[1]
    b_h5_D = B_h_D_3[2]

    E_h1_D = sp.Matrix([1, -1])
    d_h_D = sp.Matrix([T_h_max, -T_h_min])

    sys = {'A_h_D': A_h_D, 'B_h1_D': B_h1_D, 'B_h4_D': B_h4_D, 'b_h5_D': b_h5_D}
    con = {'E_h1_D': E_h1_D, 'd_h_D': d_h_D}

    return sys, con


def gen_dewh_matrices_eval_funcs():
    ts, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom = sp.symbols('ts, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom')
    T_h_min, T_h_max = sp.symbols('T_h_min, T_h_max')
    sys, con = gen_dewh_symbolic_matrices()

    A_h_D_eval = sp.lambdify((ts, C_w, A_h, U_h, m_h, D_h), sys.get('A_h_D'), "numpy", dummify=False)
    B_h1_D_eval = sp.lambdify((ts, C_w, A_h, U_h, m_h, P_h_Nom, D_h), sys.get('B_h1_D'), "numpy", dummify=False)
    B_h4_D_eval = sp.lambdify((ts, C_w, A_h, U_h, m_h, T_w, D_h), sys.get('B_h4_D'), "numpy", dummify=False)
    b_h5_D_eval = sp.lambdify((ts, C_w, A_h, U_h, m_h, T_inf, D_h), sys.get('b_h5_D'), "numpy", dummify=False)

    E_h1_D_eval = sp.lambdify((), con.get('E_h1_D'), "numpy", dummify=False)
    d_h_D_eval = sp.lambdify((T_h_min, T_h_max), con.get('d_h_D'), "numpy", dummify=False)

    return A_h_D_eval, B_h1_D_eval, B_h4_D_eval, b_h5_D_eval, E_h1_D_eval, d_h_D_eval


def gen_grid_cons_symbolic_matrices():
    P_g_min, P_g_max = sp.symbols('P_g_min, P_g_max')
    eps = sp.symbols('eps')

    E_p2 = sp.Matrix([-1, 1, 0, 0, -1, 1])
    E_p3 = sp.Matrix([-P_g_min, -P_g_max + eps, P_g_max, P_g_min, -P_g_min, P_g_max])
    E_p4 = sp.Matrix([0, 0, 1, -1, 1, -1])
    d_p = sp.Matrix([-P_g_min, -eps, 0, 0, -P_g_min, P_g_max])

    con = {'E_p2': E_p2, 'E_p3': E_p3, 'E_p4': E_p4, 'd_p': d_p}

    return con


def gen_grid_cons_eval_functions():
    P_g_min, P_g_max = sp.symbols('P_g_min, P_g_max')
    eps = sp.symbols('eps')
    con = gen_grid_cons_symbolic_matrices()

    E_p2_eval = sp.lambdify((), con.get('E_p2'), "numpy", dummify=False)
    E_p3_eval = sp.lambdify((P_g_min, P_g_max, eps), con.get('E_p3'), "numpy", dummify=False)
    E_p4_eval = sp.lambdify((), con.get('E_p4'), "numpy", dummify=False)
    d_p_eval = sp.lambdify((P_g_min, P_g_max, eps), con.get('d_p'), "numpy", dummify=False)

    return E_p2_eval, E_p3_eval, E_p4_eval, d_p_eval,


A_h_D_eval, B_h1_D_eval, B_h4_D_eval, b_h5_D_eval, E_h1_D_eval, d_h_D_eval = gen_dewh_matrices_eval_funcs()
E_p2_eval, E_p3_eval, E_p4_eval, d_p_eval = gen_grid_cons_eval_functions()

ts = 60
C_w = 4.1816 * 10 ** 3
A_h = 1
U_h = 2.7
m_h = 150
D_h = 0.018
T_w = 25
T_inf = 25
P_h_Nom = 3000
T_h_min = 40
T_h_max = 65

P_g_min = -2e6
P_g_max = 2e6
eps = np.finfo(float).eps

C_imp = 0.9  # R/kwh
C_exp = 0.1
C_imp_sub_exp = C_imp - C_exp

## INDIVIDUAL SYSTEM MATRIXES

A = A_h_D_eval(ts, C_w, A_h, U_h, m_h, D_h)
B1 = B_h1_D_eval(ts, C_w, A_h, U_h, m_h, P_h_Nom, D_h)
B2 = np.empty(0)
B3 = np.empty(0)
B4 = B_h4_D_eval(ts, C_w, A_h, U_h, m_h, T_w, D_h)
b5 = b_h5_D_eval(ts, C_w, A_h, U_h, m_h, T_inf, D_h)

E1 = E_h1_D_eval()
E2 = np.empty(0)
E3 = np.empty(0)
E4 = np.empty(0)
E5 = np.empty(0)
d = d_h_D_eval(T_h_min, T_h_max)

## POWER CONSTRAINT MATRICES

E_p2 = E_p2_eval()
E_p3 = E_p3_eval(P_g_min, P_g_max, eps)
E_p4 = E_p4_eval()
d_p = d_p_eval(P_g_min, P_g_max, eps)

# Construct Full System

Nh = 2  # Number of DewhSys's
Nb = 0  # Number of bess's
Np = 5  # Prediction horizon

## Sparse matrices - vstack fast for csr, hstack fast for csc, toeplitz method could be improved

## SPARSE COMBIMNNED SYSTEM MATRICES

As_s = scs.block_diag([A] * Nh, 'csr')
B1s_s = scs.block_diag([B1] * Nh, 'csr')
B2s_s = scs.block_diag([B2] * Nh, 'csr')
B3s_s = scs.block_diag([B3] * Nh, 'csr')
B4s_s = scs.block_diag([B4] * Nh, 'csr')
b5s_s = scs.vstack([b5] * Nh, 'csr')

E1s_s = scs.block_diag([E1] * Nh, 'csr')
E2s_s = scs.block_diag([E2] * Nh, 'csr')
E3s_s = scs.block_diag([E3] * Nh, 'csr')
E4s_s = scs.block_diag([E4] * Nh, 'csr')
E5s_s = scs.block_diag([E5] * Nh, 'csr')
ds_s = scs.vstack([scs.csc_matrix(d)] * Nh, 'csc')

## STATE Evolution Matrices

B123s_s = scs.hstack([B1s_s, B2s_s, B3s_s])

As_s_pow = [(As_s ** i) for i in range(Np + 1)]

Phi_xs_s = scs.vstack(As_s_pow)



Phi_vs_s = scs.bmat(scl.toeplitz([scs.coo_matrix(B123s_s.shape)] + [As_s_pow[i] * B123s_s for i in range(Np)],
                                 [scs.coo_matrix(B123s_s.shape)] * (Np)))

Phi_ws_s = scs.bmat(scl.toeplitz([scs.coo_matrix(B4s_s.shape)] + [As_s_pow[i] * B4s_s for i in range(Np)],
                                 [scs.coo_matrix(B4s_s.shape)] * (Np)))

Phi_b5_s = scs.bmat(scl.toeplitz([scs.coo_matrix(As_s.shape)] + [As_s_pow[i] for i in range(Np)],
                                 [scs.coo_matrix(As_s.shape)] * (Np)))

b5s_tilde_s = scs.vstack([b5s_s] * (Np))

## System Constraint Matrices

E1_tilde_s = scs.block_diag([E1s_s] * (Np + 1))
E234_tilde_s = scs.block_diag([scs.hstack([E2s_s, E3s_s, E4s_s])] * (Np + 1))
E5_tilde_s = scs.block_diag([E5s_s] * (Np + 1))
ds_tilde_s = scs.vstack([ds_s] * (Np + 1))

## Combined Constraint Matrices i.e. Hsv*Vsv <= Hs5 - Hsw*Wsv - Hsx*xs

try:
    Hsv = (E1_tilde_s + E234_tilde_s) * Phi_vs_s  # E234_tilde_s may be a null matrix
except ValueError:
    Hsv = (E1_tilde_s) * Phi_vs_s

Hs5 = (ds_tilde_s - E1_tilde_s * Phi_b5_s * b5s_tilde_s)

try:
    Hsw = E1_tilde_s * Phi_ws_s + E5_tilde_s  # E5_tilde_s may be a null matrix
except ValueError:
    Hsw = E1_tilde_s * Phi_ws_s

Hsx = E1_tilde_s * Phi_xs_s

Num_desc = Nh * Np

## Set up power constraints

N_h_P_nom = np.ones((1, Nh)) * P_h_Nom
N_b_ones = np.ones((1, Nb))
zeros_dels_zs = np.zeros((1, 0))  # NEEDS TO BE UPDATED TO INCLUDE BATTERIES ETC!!!!!
sum_load_vec = np.hstack([N_h_P_nom, N_b_ones, zeros_dels_zs])

Gam_pv = scs.block_diag([sum_load_vec] * Np)
Gam_pw = scs.block_diag([np.array([1, -1])] * Np)

E_p2_tilde = scs.block_diag([E_p2] * Np)

E_p34 = np.hstack([E_p3, E_p4])
E_p34_tilde = scs.block_diag([E_p34] * Np)

d_p_tilde = np.vstack([d_p] * Np)

H_pvs = E_p2_tilde * Gam_pv
H_pwp = E_p2_tilde * Gam_pw
H_pvp = E_p34_tilde
H_dp = d_p_tilde

## OVERALL CONSTRAINT MATRICES AND VECTORS A*V <= b ==> F1*V <= F2 + F3w*W + F4x*x

F1 = scs.bmat(np.array([[Hsv, None], [H_pvs, H_pvp]]))
F2 = scs.bmat(np.array([[Hs5], [H_dp]]))
F3w = scs.block_diag([-Hsw, -H_pvp])
F4x = scs.vstack([Hsx, scs.coo_matrix((d_p_tilde.shape[0], Hsx.shape[1]))])


## OBJECTIVE VECTOR


C_exp_tilde = np.vstack([C_exp * 1.001 ** i for i in range(Np)])
C_imp_sub_exp_tilde = np.vstack([C_imp_sub_exp * 1.001 ** i for i in range(Np)])

Cost_vec_vs = (C_exp_tilde * sum_load_vec).reshape((-1, 1))
Cost_vec_vp = (C_imp_sub_exp_tilde * np.array([[0, 1]])).reshape((-1, 1))

Cost_vec_ws = np.zeros((Hsw.shape[1], 1))
Cost_vec_wp = (C_exp_tilde * np.array([[1, -1]])).reshape((-1, 1))

Cost_vec_V_tilde = np.vstack([Cost_vec_vs, Cost_vec_vp])
Cost_vec_W_tilde = np.vstack([Cost_vec_ws, Cost_vec_wp])




################################################################
##############################  SOLVER SETUP    ##################################
################################################################

import numpy as np
import scipy as sc
import timeit
import functools

import scipy.linalg as scl
import scipy.sparse as scs
import scipy.sparse.linalg as scsl

# import pandas as pd
# import matplotlib.pylab as plt

#
# import pyomo.environ as pe
#
#
# model = pe.ConcreteModel()
#
#
# A = F1
# numvar = F1.shape[1]
# numcon = F1.shape[0]
# b = scs.csc_matrix(F2)+F4x*60
#
# index_List = [i for i in range(numvar)]
#
# model.V_index = pe.Set(initialize=index_List, ordered=True)
#
#
# def V_Var_dom(model, i):
#     if i<1000:
#         return pe.Binary
#     else:
#         return pe.Reals
#
# model.V_var = pe.Var(model.V_index, domain=V_Var_dom, initialize=0)
#
# def Mod_obj(model):
#     expr = model.V_var[2]
#     return expr
#
# model.Obj = pe.Objective(rule=Mod_obj)
#
#
# A_csr = scs.csr_matrix(A)
#
# con_Index_list = list(range(b.shape[0]))
# model.Con_index = pe.Set(initialize=con_Index_list, ordered=True)
#
#
# def Mod_con(Model,ri):
#     row = A_csr.getrow(ri)
#     if row.data.size == 0:
#         return pe.Constraint.Skip
#     else:
#         return sum(coeff*model.V_var[index] for (index, coeff) in zip(row.indices,row.data)) <= -b[ri,0]
# model.ConF1 = pe.Constraint(model.Con_index, rule=Mod_con)
#
#
# model.write("s.lp", "lp", io_options={"symbolic_solver_labels":True})
#
#
#
#
#





def fun():
    def clos():
        model.Con2 = pe.Constraint(list(range(b.shape[0])), rule=Mod_con)
        return 1

    return clos


if __name__ == '__main__':
    import timeit

    t1 = timeit.timeit(fun(), number=1)
    print(t1)
