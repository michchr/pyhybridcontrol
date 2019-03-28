from models.agents import MpcAgent
from models.micro_grid_agents import GridAgentMpc, DewhAgentMpc
from models.micro_grid_models import DewhModel, GridModel
from models.parameters import dewh_param_struct, grid_param_struct
from controllers.mpc_controller.mpc_controller import *
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from datetime import datetime as DateTime
from tools.tariff_generator import TariffGenerator

file_path = os.path.normpath(
    r"C:\Users\chris\Documents\University_on_gdrive\Thesis\Software\DHW_2_02b\DHW0001\DHW0001_DHW.txt")
#
with open(file_path, 'r') as file_object:
    line_reader = file_object.readlines()
    data = []
    for line in line_reader:
        data.append(float(line.strip()) / 60.0)

data = np.array(data)
raw_df = pd.DataFrame(data, index=pd.date_range(DateTime(2018, 12, 1), None, len(data), '1Min'), columns=['actual'])

df: pd.DataFrame = raw_df.resample('15Min').sum()

dewh_control_model = DewhModel(param_struct=dewh_param_struct, const_heat=True)
dewh_sim_model = DewhModel(param_struct=dewh_param_struct, const_heat=False)

sim_steps = 1000
N_p = 48
N_tilde = N_p + 1

omega_profile = df.values / dewh_param_struct.dt
omega_profile_hat = omega_profile[96:]
omega_profile_act = omega_profile[96:]

np.random.seed(100)
tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                             high_stnd=102.95, high_peak=339.77)

time_0 = DateTime(2018, 12, 1)
cost_profile = tariff_gen.get_price_vector(len(omega_profile), time_0, dewh_param_struct.control_dt)

dewh1 = DewhAgentMpc(N_p=N_p)
dewh2 = DewhAgentMpc(N_p=N_p)

grid = GridAgentMpc(N_p=N_p)

devices = [DewhAgentMpc(N_p=N_p) for i in range(1)]


import time
st = time.time()
for ind, dev in enumerate(devices):
    dev.x_k = 65

    start = ind*96
    end = start+N_tilde

    dev.omega_tilde_k_hat = omega_profile[start: end]
    dev.set_device_penalties(q_mu=[10e7, 10e5])
    dev.build_device()
    grid.add_device(dev)

grid.get_device_data()
grid.build_grid()
grid.mpc_controller._update_components()
grid.set_grid_device_powers()
grid.set_grid_device_costs()
grid.set_grid_device_constraints()

grid.mpc_controller.set_std_obj_weights(q_y=cost_profile[:N_tilde])
grid.mpc_controller.build()

print(f"Time to build:{time.time()-st}")
#
#
# st = time.time()
# grid.mpc_controller.solve(verbose=True, TimeLimit=200)
# print(f"Time to solve including data transfer:{time.time()-st}")
#
# x_k = []
# for dev in devices:
#     x_k.append(dev.mpc_controller.variables.x.var_N_tilde.value)
#
# x = pd.DataFrame(np.hstack(x_k))
#
# x.plot(drawstyle='steps-post')
#
#
# u_k = []
# for dev in devices:
#     u_k.append(dev.mpc_controller.variables.u.var_N_tilde.value)
#
# u = pd.DataFrame(np.hstack(u_k))
#
# u.plot(drawstyle='steps-post')