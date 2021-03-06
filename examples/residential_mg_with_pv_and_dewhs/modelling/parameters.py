from structdict import StructDict
from datetime import timedelta as TimeDelta
import numpy as np

__all__ = ['dewh_param_struct', 'grid_param_struct']

control_ts = TimeDelta(minutes=15)

#150L
dewh_param_struct = StructDict(
    C_w=4.1816 * 10 ** 3,  # J/kg/K
    A_h=2.35,  # m^2
    U_h=0.88,  # W/m^2/K
    m_h=150.0,  # kg
    T_w=15.0,  # C
    T_inf=25.0 ,  # C
    P_h_Nom=3000.0,  # W  (Joules/s)
    T_h_min=50.0,  # C
    T_h_max=65.0,  # C
    T_h_Nom=45.0, # C
    T_h_max_sub_T_h_on = 12,
    T_h_max_sub_T_h_off = 4,
    T_h = 45.0, #C
    D_h=0.0,  # kg/s
    control_ts=control_ts,
    ts=control_ts.seconds,
)

# # #250L
# dewh_param_struct = StructDict(
#     C_w=4.1816 * 10 ** 3,  # J/kg/K
#     A_h=3.22,#2.35,  # m^2
#     U_h=0.88,  # W/m^2/K
#     m_h=250,#150.0,  # kg
#     T_w=15.0,  # C
#     T_inf=25.0 ,  # C
#     P_h_Nom=4000.0,#3000.0,  # W  (Joules/s)
#     T_h_min=50.0,  # C
#     T_h_max=65.0,  # C
#     T_h_Nom=45.0, # C
#     T_h = 45.0, #C
#     D_h=0.0,  # kg/s
#     control_ts=control_ts,
#     ts=control_ts.seconds,
# )


grid_param_struct = StructDict(
    P_g_min=-2e4, # W (Joules/s)
    P_g_max=2e4, # W (Joules/s)
    eps=np.finfo(float).eps,

    control_ts = control_ts,
    ts=control_ts.seconds,
)

pv_param_struct = StructDict(
    P_pv_max = 2000, # W (Joules/s)
    P_pv_units = 1,
    control_ts = control_ts,
    ts = control_ts.seconds
)

res_demand_param_struct = StructDict(
    P_res_ave = 1200, # W (Joules/s)
    P_res_units = 1,
    control_ts = control_ts,
    ts = control_ts.seconds
)
