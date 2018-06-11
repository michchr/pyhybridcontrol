
import functools
import sympy as sp

import struct_dict as sd

def _get_syms_tup(expr):
    sym_dict = {str(sym): sym for sym in expr.free_symbols}
    sym_str_list = sorted(sym_dict.keys())
    sym_list = [sym_dict.get(sym_str) for sym_str in sym_str_list]
    return tuple(sym_list), tuple(sym_str_list)

def _get_all_syms_as_str_list(*args):
    sym_dict = {str(sym): sym for dict_i in args for key, expr in dict_i.items() for sym in expr.free_symbols}
    sym_str_list = sorted(sym_dict.keys())
    return sym_str_list

def _lam_wrapper(func, local_syms_str):
    @functools.wraps(func)
    def wrapped(param_dict):
        arg_list = [param_dict.get(sym_str) for sym_str in local_syms_str]
        if None in arg_list:
            raise KeyError("Incorrect parameter supplied, requires dict with keys %s" % list(local_syms_str))
        return func(*arg_list)

    return wrapped

def _get_expr_dim(expr, is_con = False):
    if expr == None:
        return 0
    elif expr.is_Matrix:
        n,m = expr.shape
        if is_con:
            return n
        else:
            return m
    elif expr.is_Function or expr.is_algebraic or expr.is_Mul:
        return 1
    else:
        raise ValueError("Invalid Expression: expr = {}".format(expr))


class dewh_model_generator():
    def __init__(self, const_heat=True):
        eval_func_ret = self.gen_dewh_mld_sys_matrix_eval_funcs(const_heat=const_heat)
        self.sys_matrix_eval_dict = eval_func_ret[0]
        self.con_matrix_eval_dict = eval_func_ret[1]
        self.var_dim_struct = eval_func_ret[2]
        self.required_param_list = eval_func_ret[3]

    def gen_dewh_mld_sys_matrix_eval_funcs(self, const_heat=True):
        sys_sym_dict, con_sym_dict, var_dim_struct = self.gen_dewh_symbolic_mld_sys_matrices(const_heat=const_heat)
        required_model_params = _get_all_syms_as_str_list(sys_sym_dict, con_sym_dict)

        sys_dict_eval = {}
        for key, expr in sys_sym_dict.items():
            syms_tup, syms_str_tup = _get_syms_tup(expr)
            lam = sp.lambdify(syms_tup, expr, "numpy", dummify=False)
            sys_dict_eval[key + "_eval"] = _lam_wrapper(lam, syms_str_tup)
            # sys_dict_eval[key + "_eval2"] = sp.lambdify(syms_tup, expr, "numpy", dummify=False)

        con_dict_eval = {}
        for key, expr in con_sym_dict.items():
            syms_tup, syms_str_tup = _get_syms_tup(expr)
            lam = sp.lambdify(syms_tup, expr, "numpy", dummify=False)
            con_dict_eval[key + "_eval"] = _lam_wrapper(lam, syms_str_tup)
            # con_dict_eval[key + "_eval2"] = sp.lambdify(syms_tup, expr, "numpy", dummify=False)

        return sys_dict_eval, con_dict_eval, var_dim_struct, required_model_params

    def gen_dewh_symbolic_mld_sys_matrices(self, default_time_units = 'sec', const_heat=True):
        ts, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom = sp.symbols('ts, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom')
        T_h_min, T_h_max = sp.symbols('T_h_min, T_h_max')

        if default_time_units == 'sec':
            divisor = 1
        elif default_time_units == 'min':
            divisor = 60
        elif default_time_units == 'hour':
            divisor = 3600
        else:
            raise ValueError("default_time_units = %s, not a valid argument" %default_time_units)

        p1 = U_h*divisor * A_h
        p2 = m_h * C_w

        # Define continuous system matrices
        if const_heat:  # assume heat demand constant over sampling period
            A_h_c = -p1 / p2
            B_h_c = sp.Matrix([[P_h_Nom*divisor, -1, p1 * T_inf]]) * (p2 ** -1)
        else:  # assume water demand volume constant over sampling period
            A_h_c = -(D_h/divisor * C_w + p1) / p2
            B_h_c = sp.Matrix([[P_h_Nom*divisor, C_w * T_w, p1 * T_inf]]) * (p2 ** -1)

        # Compute discretized system matrices
        A_h = sp.exp(A_h_c * ts)
        B_h_3 = A_h_c ** (-1) * (sp.exp(A_h_c * ts) - 1) * B_h_c

        sys_sym_dict = {}
        sys_sym_dict['A_h'] = A_h
        sys_sym_dict['B_h1'] = B_h_3[0]
        sys_sym_dict['B_h4'] = B_h_3[1]
        sys_sym_dict['b_h5'] = B_h_3[2]

        cons_sym_dict = {}
        cons_sym_dict['E_h1'] = sp.Matrix([1, -1])
        cons_sym_dict['d_h'] = sp.Matrix([T_h_max, -T_h_min])

        vardim_struct = sd.Struct_Dict(
            nstates = max(_get_expr_dim(sys_sym_dict.get('A_h')),_get_expr_dim(cons_sym_dict.get('E_h1'))),
            ncons = _get_expr_dim(cons_sym_dict.get('d_h'), is_con=True),
            nx = max(_get_expr_dim(sys_sym_dict.get('A_h')),_get_expr_dim(cons_sym_dict.get('E_h1'))),
            nu = max(_get_expr_dim(sys_sym_dict.get('B_h1')),_get_expr_dim(cons_sym_dict.get('E_h2'))),
            ndelta = max(_get_expr_dim(sys_sym_dict.get('B_h2')),_get_expr_dim(cons_sym_dict.get('E_h3'))),
            nz = max(_get_expr_dim(sys_sym_dict.get('B_h3')),_get_expr_dim(cons_sym_dict.get('E_h4'))),
            nomega = max(_get_expr_dim(sys_sym_dict.get('B_h4')),_get_expr_dim(cons_sym_dict.get('E_h5'))),
            nx_l = 0, # number of binary states
            nu_l = 1, # number of binary inputs
            nomega_l = 0, # number of binary disturbances
        )

        return sys_sym_dict, cons_sym_dict, vardim_struct

class grid_model_generator():
    def __init__(self):
        eval_func_ret = self.gen_grid_mld_cons_matrix_eval_funcs()
        self.con_matrix_eval_dict = eval_func_ret[0]
        self.required_param_list = eval_func_ret[1]

    def gen_grid_mld_cons_matrix_eval_funcs(self):
        cons_sym_dict = self.gen_grid_symbolic_mld_cons_matrices()
        required_model_params = _get_all_syms_as_str_list(cons_sym_dict)

        cons_dict_eval = {}
        for key, expr in cons_sym_dict.items():
            syms_tup, syms_str_tup = _get_syms_tup(expr)
            lam = sp.lambdify(syms_tup, expr, "numpy", dummify=False)
            cons_dict_eval[key + "_eval"] = _lam_wrapper(lam, syms_str_tup)
            # cons_dict_eval[key + "_eval2"] = sp.lambdify(syms_tup, expr, "numpy", dummify=False)

        return cons_dict_eval, required_model_params

    def gen_grid_symbolic_mld_cons_matrices(self):
        P_g_min, P_g_max = sp.symbols('P_g_min, P_g_max')
        eps = sp.symbols('eps')

        cons_dict = {}

        cons_dict['E_p2'] = sp.Matrix([-1, 1, 0, 0, -1, 1])
        cons_dict['E_p3'] = sp.Matrix([-P_g_min, -P_g_max + eps, P_g_max, P_g_min, -P_g_min, P_g_max])
        cons_dict['E_p4'] = sp.Matrix([0, 0, 1, -1, 1, -1])
        cons_dict['d_p'] = sp.Matrix([-P_g_min, -eps, 0, 0, -P_g_min, P_g_max])

        return cons_dict


if __name__ == '__main__':
    import pprint
    import numpy as np
    import timeit

    control_ts = 60

    dewh_p = sd.Struct_Dict()
    dewh_p.C_w = 4.1816 * 10 ** 3
    dewh_p.A_h = 1
    dewh_p.U_h = 2.7
    dewh_p.m_h = 150
    dewh_p.T_w = 25
    dewh_p.T_inf = 25
    dewh_p.P_h_Nom = 3000
    dewh_p.T_h_min = 40
    dewh_p.T_h_max = 65
    dewh_p.ts = control_ts

    grid_p = sd.Struct_Dict()
    grid_p.P_g_min = -2e6
    grid_p.P_g_max = 2e6
    grid_p.eps = np.finfo(float).eps

    grid_p.C_imp = 0.9  # R/kwh
    grid_p.C_exp = 0.1
    grid_p.C_imp_sub_exp = grid_p.C_imp - grid_p.C_exp

    dewh_gen = dewh_model_generator()
    grid_gen = grid_model_generator()
    def func():
        def closure():
            c = dewh_gen.con_matrix_eval_dict.get("E_h1_eval")
            print(c(dewh_p))
            print(dewh_gen.var_dim_struct)
            return 1
        return closure

    t1 = timeit.timeit(func(), number=1)
    print(t1)
