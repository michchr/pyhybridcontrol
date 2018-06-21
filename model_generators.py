import functools
import sympy as sp

from structdict import StructDict


class DewhModelGenerator():
    def __init__(self, const_heat=True):
        eval_func_ret = self.gen_dewh_mld_sys_matrix_eval_funcs(const_heat=const_heat)
        self.mld_eval_struct = eval_func_ret[0]
        self.var_dim_struct = eval_func_ret[1]
        self.required_param_list = eval_func_ret[2]

    def gen_dewh_mld_sys_matrix_eval_funcs(self, const_heat=True):
        mld_sym_struct, var_dim_struct = self.gen_dewh_symbolic_mld_sys_matrices(const_heat=const_heat)
        required_model_params = _get_all_syms_as_str_list(mld_sym_struct)

        mld_eval_struct = StructDict()
        for key, expr in mld_sym_struct.items():
            syms_tup, syms_str_tup = _get_syms_tup(expr)
            lam = sp.lambdify(syms_tup, expr, "numpy", dummify=False)
            mld_eval_struct[key + "_eval"] = _lam_wrapper(lam, syms_str_tup)
            # mld_eval_struct[key + "_eval2"] = sp.lambdify(syms_tup, expr, "numpy", dummify=False)

        return mld_eval_struct, var_dim_struct, required_model_params

    def gen_dewh_symbolic_mld_sys_matrices(self, default_time_units='sec', const_heat=True):
        ts, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom = sp.symbols(
            'ts, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom')
        T_h_min, T_h_max = sp.symbols('T_h_min, T_h_max')

        p1 = U_h * A_h
        p2 = m_h * C_w

        # Define continuous system matrices
        if const_heat:  # assume heat demand constant over sampling period
            A_h_c = -p1 / p2
            B_h_c = sp.Matrix([[P_h_Nom, -1, p1 * T_inf]]) * (p2 ** -1)
        else:  # assume water demand volume constant over sampling period
            A_h_c = -(D_h * C_w + p1) / p2
            B_h_c = sp.Matrix([[P_h_Nom, C_w * T_w, p1 * T_inf]]) * (p2 ** -1)

        # Compute discretized system matrices
        A_h = sp.exp(A_h_c * ts)
        B_h_3 = A_h_c ** (-1) * (sp.exp(A_h_c * ts) - 1) * B_h_c

        mld_sym_struct = StructDict()
        mld_sym_struct.A_h = A_h
        mld_sym_struct.B_h1 = B_h_3[0]
        mld_sym_struct.B_h4 = B_h_3[1]
        mld_sym_struct.b_h5 = B_h_3[2]

        mld_sym_struct.E_h1 = sp.Matrix([1, -1])
        mld_sym_struct.d_h = sp.Matrix([T_h_max, -T_h_min])

        mld_sym_struct.B_h2 = sp.Matrix([])
        mld_sym_struct.B_h3 = sp.Matrix([])

        mld_sym_struct.E_h2 = sp.Matrix([])
        mld_sym_struct.E_h3 = sp.Matrix([])
        mld_sym_struct.E_h4 = sp.Matrix([])
        mld_sym_struct.E_h5 = sp.Matrix([])

        vardim_struct = StructDict(
            nstates=max(_get_expr_dim(mld_sym_struct.A_h), _get_expr_dim(mld_sym_struct.E_h1)),
            ncons=_get_expr_dim(mld_sym_struct.d_h, is_con=True),
            nx=max(_get_expr_dim(mld_sym_struct.A_h), _get_expr_dim(mld_sym_struct.E_h1)),
            nu=max(_get_expr_dim(mld_sym_struct.B_h1), _get_expr_dim(mld_sym_struct.E_h2)),
            ndelta=max(_get_expr_dim(mld_sym_struct.B_h2), _get_expr_dim(mld_sym_struct.E_h3)),
            nz=max(_get_expr_dim(mld_sym_struct.B_h3), _get_expr_dim(mld_sym_struct.E_h4)),
            nomega=max(_get_expr_dim(mld_sym_struct.B_h4), _get_expr_dim(mld_sym_struct.E_h5)),
            nx_l=0,  # number of binary states
            nu_l=1,  # number of binary inputs
            nomega_l=0,  # number of binary disturbances
        )

        return mld_sym_struct, vardim_struct


# END_CLASS DewhModelGenerator


class GridModelGenerator():
    def __init__(self):
        eval_func_ret = self.gen_grid_mld_cons_matrix_eval_funcs()
        self.mld_eval_struct = eval_func_ret[0]
        self.required_param_list = eval_func_ret[1]

    def gen_grid_mld_cons_matrix_eval_funcs(self):
        mld_sym_struct = self.gen_grid_symbolic_mld_cons_matrices()
        required_model_params = _get_all_syms_as_str_list(mld_sym_struct)

        mld_eval_struct = StructDict()
        for key, expr in mld_sym_struct.items():
            syms_tup, syms_str_tup = _get_syms_tup(expr)
            lam = sp.lambdify(syms_tup, expr, "numpy", dummify=False)
            mld_eval_struct[key + "_eval"] = _lam_wrapper(lam, syms_str_tup)
            # cons_dict_eval[key + "_eval2"] = sp.lambdify(syms_tup, expr, "numpy", dummify=False)

        return mld_eval_struct, required_model_params

    def gen_grid_symbolic_mld_cons_matrices(self):
        P_g_min, P_g_max = sp.symbols('P_g_min, P_g_max')
        eps = sp.symbols('eps')

        mld_sym_struct = StructDict()

        mld_sym_struct.E_p2 = sp.Matrix([-1, 1, 0, 0, -1, 1])
        mld_sym_struct.E_p3 = sp.Matrix([-P_g_min, -P_g_max + eps, P_g_max, P_g_min, -P_g_min, P_g_max])
        mld_sym_struct.E_p4 = sp.Matrix([0, 0, 1, -1, 1, -1])
        mld_sym_struct.d_p = sp.Matrix([-P_g_min, -eps, 0, 0, -P_g_min, P_g_max])

        mld_sym_struct.A_p = sp.Matrix([])
        mld_sym_struct.B_p1 = sp.Matrix([])
        mld_sym_struct.B_p2 = sp.Matrix([])
        mld_sym_struct.B_p3 = sp.Matrix([])
        mld_sym_struct.B_p4 = sp.Matrix([])
        mld_sym_struct.b_p5 = sp.Matrix([])

        mld_sym_struct.E_p1 = sp.Matrix([])
        mld_sym_struct.E_p5 = sp.Matrix([])

        return mld_sym_struct


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
            raise KeyError("Incorrect parameter supplied, requires dict with keys {}".format(local_syms_str))
        return func(*arg_list)

    return wrapped


def _get_expr_dim(expr, is_con=False):
    if expr == None:
        return 0
    elif expr.is_Matrix:
        n, m = expr.shape
        if is_con:
            return n
        else:
            return m
    elif expr.is_Function or expr.is_algebraic or expr.is_Mul:
        return 1
    else:
        raise ValueError("Invalid Expression: expr = {}".format(expr))


if __name__ == '__main__':
    import pprint
    import numpy as np
    import timeit

    from parameters import dewh_p, grid_p

    dewh_gen = DewhModelGenerator()
    grid_gen = GridModelGenerator()


    def func():
        def closure():
            c = dewh_gen.mld_eval_struct.get("E_h3_eval")
            print(c(dewh_p))
            print(dewh_gen.var_dim_struct)
            return 1

        return closure


    t1 = timeit.timeit(func(), number=1)
    print(t1)
