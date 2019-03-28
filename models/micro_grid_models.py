import numpy as np
import sympy as sp

from models.mld_model import PvMldSystemModel, MldModel
from models.parameters import dewh_param_struct, grid_param_struct, pv_param_struct, res_demand_param_struct
from structdict import StructDict
from utils.decorator_utils import cache_hashable_args
from utils.helper_funcs import is_all_None


class DewhModel(PvMldSystemModel):
    def __init__(self, param_struct=None, const_heat=True,
                 mld_numeric=None, mld_callable=None, mld_symbolic=None):

        param_struct = param_struct or dewh_param_struct

        if is_all_None(mld_numeric, mld_callable, mld_symbolic):
            mld_symbolic = self.get_dewh_mld_symbolic(const_heat=const_heat)

        super(DewhModel, self).__init__(mld_numeric=mld_numeric,
                                        mld_symbolic=mld_symbolic,
                                        mld_callable=mld_callable,
                                        param_struct=param_struct)

    @staticmethod
    @cache_hashable_args(maxsize=2)
    def get_dewh_mld_symbolic(const_heat=True):
        dt, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom = sp.symbols(
            'dt, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom')
        T_h_min, T_h_max = sp.symbols('T_h_min, T_h_max')
        T_h_Nom = sp.symbols('T_h_Nom')
        T_h = sp.symbols('T_h')

        p1 = U_h * A_h
        p2 = m_h * C_w
        # Define continuous system matrices
        if const_heat:
            # Assume heat demand constant over sampling period and that
            # that energy demand is equivalent to energy of water volume
            # extracted at T_h_Nom
            A_c = sp.Matrix([-p1]) * (p2 ** -1)
            B1_c = sp.Matrix([P_h_Nom]) * (p2 ** -1)
            B4_c = sp.Matrix([C_w * (T_w - T_h_Nom)]) * (p2 ** -1)
            b5_c = sp.Matrix([p1 * T_inf]) * (p2 ** -1)
        else:
            # assume water demand flow rate constant over sampling period
            A_c = sp.Matrix([-((D_h * (T_h_Nom - T_w) / (T_h - T_w) * C_w) + p1) / p2])
            B1_c = sp.Matrix([P_h_Nom]) * (p2 ** -1)
            B4_c = sp.Matrix([C_w * T_w * (T_h_Nom - T_w) / (T_h - T_w)]) * (p2 ** -1)
            b5_c = sp.Matrix([p1 * T_inf]) * (p2 ** -1)

        # Compute discretized system matrices
        A = sp.Matrix.exp(A_c * dt)
        em = A_c.pinv() * (sp.Matrix.exp(A_c * dt) - sp.eye(*A.shape))
        B1 = em * B1_c
        B4 = em * B4_c
        b5 = em * b5_c

        mld_sym_struct = StructDict()
        mld_sym_struct.A = A
        mld_sym_struct.B1 = B1
        mld_sym_struct.B4 = B4
        mld_sym_struct.b5 = b5

        mld_sym_struct.E = np.array([[1,
                                      -1,
                                      0,
                                      0]]).T
        mld_sym_struct.F1 = np.array([[0,
                                       0,
                                       1,
                                       -1]]).T

        mld_sym_struct.Psi = np.array([[-1, 0],
                                       [0, -1],
                                       [0, 0],
                                       [0, 0]])

        mld_sym_struct.f5 = sp.Matrix([[T_h_max,
                                        -T_h_min,
                                        1,
                                        0]]).T

        MldModel_sym = MldModel(mld_sym_struct, nu_l=1, dt=0)

        return MldModel_sym


class GridModel(PvMldSystemModel):
    def __init__(self, param_struct=None, num_devices=None,
                 mld_numeric=None, mld_callable=None, mld_symbolic=None):

        param_struct = param_struct or grid_param_struct

        num_devices = num_devices if num_devices is not None else 0

        if not np.issubdtype(type(num_devices), np.integer):
            raise ValueError("num_devices must be an integer")
        else:
            self._num_devices = num_devices

        if is_all_None(mld_numeric, mld_callable, mld_symbolic):
            mld_symbolic = self.get_grid_mld_symbolic(num_devices=num_devices)

        super(GridModel, self).__init__(mld_numeric=mld_numeric,
                                        mld_symbolic=mld_symbolic,
                                        mld_callable=mld_callable,
                                        param_struct=param_struct)

    @property
    def num_devices(self):
        return self._num_devices

    @num_devices.setter
    def num_devices(self, num_devices):
        if num_devices != self._num_devices:
            mld_symbolic = self.get_grid_mld_symbolic(num_devices)
            self.update_mld(mld_symbolic=mld_symbolic)
            self._num_devices = num_devices

    @staticmethod
    def get_grid_mld_symbolic(num_devices):
        P_g_min, P_g_max = sp.symbols('P_g_min, P_g_max')
        eps = sp.symbols('eps')

        mld_sym_struct = StructDict()

        mld_sym_struct.D4 = np.ones((1, num_devices))

        mld_sym_struct.F2 = sp.Matrix([[-P_g_min,
                                        -(P_g_max + eps),
                                        -P_g_max,
                                        P_g_min,
                                        -P_g_min,
                                        P_g_max]]).T
        mld_sym_struct.F3 = sp.Matrix([[0,
                                        0,
                                        1,
                                        -1,
                                        1,
                                        -1]]).T
        mld_sym_struct.f5 = sp.Matrix([[-P_g_min,
                                        -eps,
                                        0,
                                        0,
                                        -P_g_min,
                                        P_g_max]]).T
        mld_sym_struct.G = sp.Matrix([[-1,
                                       1,
                                       0,
                                       0,
                                       -1,
                                       1]]).T

        MldModel_sym = MldModel(mld_sym_struct, dt=0)

        return MldModel_sym


class PvModel(PvMldSystemModel):
    def __init__(self, param_struct=None,
                 mld_numeric=None, mld_callable=None, mld_symbolic=None):

        param_struct = param_struct or pv_param_struct


        if is_all_None(mld_numeric, mld_callable, mld_symbolic):
            mld_symbolic = self.get_pv_mld_symbolic()

        super(PvModel, self).__init__(mld_numeric=mld_numeric,
                                      mld_symbolic=mld_symbolic,
                                      mld_callable=mld_callable,
                                      param_struct=param_struct)

    @staticmethod
    def get_pv_mld_symbolic():
        P_pv_max = sp.symbols('P_pv_max')
        P_pv_units = sp.symbols('P_pv_units')

        mld_sym_struct = StructDict()

        mld_sym_struct.D4 = sp.Matrix([[P_pv_max*P_pv_units]])

        MldModel_sym = MldModel(mld_sym_struct, dt=0)


        return MldModel_sym


class ResDemandModel(PvMldSystemModel):
    def __init__(self, param_struct=None,
                 mld_numeric=None, mld_callable=None, mld_symbolic=None):

        param_struct = param_struct or res_demand_param_struct

        if is_all_None(mld_numeric, mld_callable, mld_symbolic):
            mld_symbolic = self.get_res_demand_mld_symbolic()

        super(ResDemandModel, self).__init__(mld_numeric=mld_numeric,
                                      mld_symbolic=mld_symbolic,
                                      mld_callable=mld_callable,
                                      param_struct=param_struct)

    @staticmethod
    def get_res_demand_mld_symbolic():
        P_res_ave = sp.symbols('P_res_ave')
        P_res_units = sp.symbols('P_res_units')

        mld_sym_struct = StructDict()

        mld_sym_struct.D4 = sp.Matrix([[P_res_ave*P_res_units]])

        MldModel_sym = MldModel(mld_sym_struct, dt=0)

        return MldModel_sym