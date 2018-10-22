import sympy as sp
import pandas as pd

from utils.structdict import StructDict
from models.mld_model import MldModel

from tools.grid_dataframe import MicroGridDataFrame, MicroGridSeries, IDX
from tools.mongo_interface import MongoInterface

pd.set_option('mode.chained_assignment', 'raise')



class DeviceModelGenerator:

    def __init__(self, *args, **kwargs):
        self.symbolic_mld = self.get_symbolic_mld(*args, **kwargs)
        self.eval_mld = self.get_eval_mld()

    def get_symbolic_mld(self):
        raise NotImplementedError("Need to implement symbolic mld")

    def get_eval_mld(self, symbolic_mld=None):
        symbolic_mld = symbolic_mld or self.symbolic_mld
        return symbolic_mld.to_eval()

    def get_numeric_mld(self, param_struct=None, mld=None):
        mld = mld or self.eval_mld or self.symbolic_mld
        return mld.to_numeric(param_struct=param_struct)

    def get_required_params(self):
        return self.symbolic_mld._get_all_syms_str_list()


class DewhModelGenerator(DeviceModelGenerator):
    def __init__(self, *args, const_heat=True, **kwargs):
        super(DewhModelGenerator, self).__init__(*args, const_heat=const_heat, **kwargs)

    def get_symbolic_mld(self, const_heat=True):
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
        mld_sym_struct.A = A_h
        mld_sym_struct.B1 = B_h_3[0]
        mld_sym_struct.B4 = B_h_3[1]
        mld_sym_struct.b5 = B_h_3[2]

        mld_sym_struct.E1 = sp.Matrix([1, -1])
        mld_sym_struct.d = sp.Matrix([T_h_max, -T_h_min])

        MldModel_sym = MldModel(mld_sym_struct, nu_l=1)

        return MldModel_sym


class GridModelGenerator(DeviceModelGenerator):

    def get_symbolic_mld(self):
        P_g_min, P_g_max = sp.symbols('P_g_min, P_g_max')
        eps = sp.symbols('eps')

        mld_sym_struct = StructDict()

        mld_sym_struct.E2 = sp.Matrix([-1, 1, 0, 0, -1, 1])
        mld_sym_struct.E3 = sp.Matrix([-P_g_min, -(P_g_max + eps), -P_g_max, P_g_min, -P_g_min, P_g_max])
        mld_sym_struct.E4 = sp.Matrix([0, 0, 1, -1, 1, -1])
        mld_sym_struct.d = sp.Matrix([-P_g_min, -eps, 0, 0, -P_g_min, P_g_max])

        MldModel_sym = MldModel(mld_sym_struct, nu_l=0, nx_l=0, nomega_l=0)

        return MldModel_sym


class Dewh:
    def __init__(self, device_model_generator: DeviceModelGenerator, dev_id=None, param_struct=None):
        self.device_model_generator = device_model_generator
        self.mld = self.device_model_generator.get_numeric_mld(param_struct=param_struct)
        self.dev_id = dev_id
        self.device_type = 'dewh'
        self.historical_df = MicroGridDataFrame()

    @property
    def symbolic_mld(self):
        return self.device_model_generator.symbolic_mld

    def load_historical(self, start_datetime=None, end_datetime=None):
        mi = MongoInterface(database='site_data', collection='Kikuyu')
        raw_data = mi.get_one_dev_raw_dataframe(self.device_type, self.dev_id, start_datetime=start_datetime,
                                                end_datetime=end_datetime)
        mi.close()

        df = raw_data.resample_device('15Min')
        self.historical_df = df
        return  df

if __name__ == '__main__':
    from datetime import datetime as DateTime
    from models.parameters import dewh_p

    start_datetime = DateTime(2018,8,1)
    end_datetime = DateTime(2018, 8, 20)

    dewh_g = DewhModelGenerator()

    t_dewh = Dewh(dewh_g, dev_id=1, param_struct=dewh_p)

    df = t_dewh.load_historical(start_datetime, end_datetime)

    from matplotlib import pyplot as plt
    df.stair_plot(style='.-')
    plt.show()