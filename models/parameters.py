from utils.structdict import StructDict
from datetime import timedelta as TimeDelta
import numpy as np


__all__ = ['dewh_p', 'grid_p']

control_ts = TimeDelta(minutes=3)

dewh_p = StructDict()
dewh_p.C_w = 4.1816 * 10 ** 3   #J/kg/K
dewh_p.A_h = 2.35       #m^2
dewh_p.U_h = 0.88    #W/m^2/K
dewh_p.m_h = 150.0      #kg
dewh_p.T_w = 15.0       #K
dewh_p.T_inf = 25.0    #K
dewh_p.P_h_Nom = 3000.0 #W  (Joules/s)
dewh_p.T_h_min = 40.0   #K
dewh_p.T_h_max = 65.0   #K
dewh_p.D_h = 0.0        #kg/s
dewh_p.control_ts = control_ts
dewh_p.ts = control_ts.seconds


grid_p = StructDict()
grid_p.P_g_min = -2e6
grid_p.P_g_max = 2e6
grid_p.eps = np.finfo(float).eps

grid_p.C_imp = None  # R/kwh
grid_p.C_exp = None
grid_p.C_imp_sub_exp = grid_p.C_imp - grid_p.C_exp if grid_p.C_imp and grid_p.C_exp else None

grid_p.ts = control_ts.seconds
