from structdict import StructDict
import numpy as np

__all__ = ['dewh_p', 'grid_p']

control_ts = 15 * 60

dewh_p = StructDict()
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

grid_p = StructDict()
grid_p.P_g_min = -2e4
grid_p.P_g_max = 2e4
grid_p.eps = np.finfo(float).eps

grid_p.C_imp = None  # R/kwh
grid_p.C_exp = None
grid_p.C_imp_sub_exp = grid_p.C_imp - grid_p.C_exp if grid_p.C_imp and grid_p.C_exp else None

grid_p.ts = control_ts
