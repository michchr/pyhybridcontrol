from structdict import StructDict
from datetime import timedelta as TimeDelta
import numpy as np

__all__ = ['dewh_p', 'grid_p']

control_dt = TimeDelta(minutes=15)


dewh_p = StructDict(
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
    T_h = 0.0, #C
    D_h=0.0,  # kg/s
    control_dt=control_dt,
    dt=control_dt.seconds,
)

C_imp = None
C_exp = None

grid_p = StructDict(
    num_devices = 3,

    P_g_min=-2e6,
    P_g_max=2e6,
    eps=np.finfo(float).eps,

    C_imp=None,  # R/kwh
    C_exp=None,
    C_imp_sub_exp=C_imp - C_exp if C_imp and C_exp else None,

    dt=control_dt.seconds,
)
