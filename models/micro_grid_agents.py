import sympy as sp
import pandas as pd
import numpy as np
import scipy.linalg as scl

from utils.structdict import StructDict
from models.mld_model import MldModel
from models.agents import AgentModelGenerator, Agent

from copy import copy as _copy, deepcopy as _deepcopy

from tools.grid_dataframe import MicroGridDataFrame, MicroGridSeries, IDX
from tools.mongo_interface import MongoInterface


class DewhModelGenerator(AgentModelGenerator):
    def __init__(self, *args, const_heat=True, **kwargs):
        super(DewhModelGenerator, self).__init__(*args, const_heat=const_heat, **kwargs)

    def get_mld_symbolic(self, const_heat=True):
        dt, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom = sp.symbols(
            'dt, C_w, A_h, U_h, m_h, D_h, T_w, T_inf, P_h_Nom')
        T_h_min, T_h_max = sp.symbols('T_h_min, T_h_max')

        p1 = U_h * A_h
        p2 = m_h * C_w

        # Define continuous system matrices
        if const_heat:  # assume heat demand constant over sampling period
            A_c = sp.Matrix([-p1 / p2])
            B1_c = sp.Matrix([P_h_Nom]) * (p2 ** -1)
            B4_c = sp.Matrix([-1]) * (p2 ** -1)
            b5_c = sp.Matrix([p1 * T_inf]) * (p2 ** -1)
        else:  # assume water demand flow rate constant over sampling period
            A_c = sp.Matrix([-(D_h * C_w + p1) / p2])
            B1_c = sp.Matrix([P_h_Nom]) * (p2 ** -1)
            B4_c = sp.Matrix([C_w * T_w]) * (p2 ** -1)
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

        mld_sym_struct.E1 = sp.Matrix([1, -1])
        mld_sym_struct.g6 = sp.Matrix([T_h_max, -T_h_min])

        MldModel_sym = MldModel(mld_sym_struct, nu_l=1)

        return MldModel_sym


class GridModelGenerator(AgentModelGenerator):

    def get_mld_symbolic(self):
        P_g_min, P_g_max = sp.symbols('P_g_min, P_g_max')
        eps = sp.symbols('eps')

        mld_sym_struct = StructDict()

        mld_sym_struct.E2 = sp.Matrix([-1, 1, 0, 0, -1, 1])
        mld_sym_struct.E3 = sp.Matrix([-P_g_min, -(P_g_max + eps), -P_g_max, P_g_min, -P_g_min, P_g_max])
        mld_sym_struct.E4 = sp.Matrix([0, 0, 1, -1, 1, -1])
        mld_sym_struct.g6 = sp.Matrix([-P_g_min, -eps, 0, 0, -P_g_min, P_g_max])

        MldModel_sym = MldModel(mld_sym_struct, nu_l=0, nx_l=0, nomega_l=0)

        return MldModel_sym

class Dewh:
    def __init__(self, device_model_generator: AgentModelGenerator, dev_id=None, param_struct=None):
        self.device_model_generator = device_model_generator
        self.mld: MldModel = self.device_model_generator.get_mld_numeric(param_struct=param_struct)
        self.param_struct = param_struct
        self.dev_id = dev_id
        self.device_type = 'dewh'
        self.historical_df = MicroGridDataFrame()

    def __getattr__(self, item):
        try:
            return self.mld[item]
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, item))

    @property
    def symbolic_mld(self):
        return self.device_model_generator.mld_symbolic

    def load_historical(self, start_datetime=None, end_datetime=None):
        with MongoInterface(database='site_data', collection='Kikuyu') as mi:
            raw_data = mi.get_one_dev_raw_dataframe(self.device_type, self.dev_id, start_datetime=start_datetime,
                                                    end_datetime=end_datetime)

        df = raw_data.resample_device(self.param_struct.control_dt)
        self.historical_df = df
        return df

    def compute_historical_demand(self):
        x_k = self.historical_df.loc[:, IDX[self.dev_id, "Temp"]].values_2d.T
        x_k_neg1 = self.historical_df.loc[:, IDX[self.dev_id, "Temp"]].shift(1).values_2d.T
        u_k_neg1 = self.historical_df.loc[:, IDX[self.dev_id, "Status"]].shift(1).values_2d.T

        omega_k_neg1 = (scl.pinv(self.B4) @ (x_k - self.A @ x_k_neg1 - self.B1 @ u_k_neg1 - self.b5))

        omega_k = pd.DataFrame(omega_k_neg1.T, index=self.historical_df.index).shift(-1).fillna(0)
        self.historical_df.loc[:, IDX[self.dev_id, 'Demand']] = omega_k.values

    def lsim(self):
        u_k = self.historical_df.loc[:, IDX[self.dev_id, "Status"]].values_2d
        x_0 = (self.historical_df.loc[:, IDX[self.dev_id, "Temp"]].values_2d)[0][0]
        omega_k = self.historical_df.loc[:, IDX[self.dev_id, 'Demand']].values_2d

        sim_out = self.mld.lsim(u_k, None, None, omega_k, x_0)

        self.historical_df.loc[:, IDX[self.dev_id, 'X_k']] = sim_out['x_out']
        return sim_out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from datetime import datetime as DateTime
    from models.parameters import dewh_p

    start_datetime = DateTime(2018, 8, 26)
    end_datetime = DateTime(2018, 8, 30)

    dewh_g = DewhModelGenerator(const_heat=True)
    for dev_id in range(3, 4):
        dewh = Dewh(dewh_g, dev_id=dev_id, param_struct=dewh_p)
        df = dewh.load_historical(start_datetime, end_datetime)
        dewh.compute_historical_demand()
        dewh.lsim()
        print(dewh.historical_df.head())
        print(np.mean(np.abs(dewh.historical_df[dev_id].Temp - dewh.historical_df[dev_id].X_k)))
        df_plt = dewh.historical_df.loc[:, IDX[dev_id, ['Temp', 'Demand', 'Status']]].resample_device('15Min')
        df_plt.stair_plot(subplots=True)

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

    # with MongoInterface(database='site_data', collection='Kikuyu') as mi:
    #     raw_data = mi.get_one_dev_raw_dataframe('dewh', 99, start_datetime=start_datetime,
    #                                             end_datetime=end_datetime)
