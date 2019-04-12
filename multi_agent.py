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

np.random.seed(100)
tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                             high_stnd=102.95, high_peak=339.77)

time_0 = DateTime(2018, 12, 1)
cost_profile = tariff_gen.get_price_vector(time_0, len(omega_profile), dewh_param_struct.control_dt)

dewh1 = DewhAgentMpc(N_p=N_p)
dewh2 = DewhAgentMpc(N_p=N_p)

grid = GridAgentMpc(N_p=N_p)

import time

st = time.time()

devices = [DewhAgentMpc(N_p=N_p) for i in range(2)]

print(f"Time to create dewh's':{time.time() - st}")


st = time.time()
omega_profile_hstack = omega_profile.reshape(96, -1, order='F')

for ind, dev in enumerate(devices):
    dev.x_k = np.random.randint(55, 65)

    start = ind * 96
    end = start + N_tilde

    dev.set_device_objective_atoms(q_mu=np.array(np.random.uniform(0.8, 1)) * [10e9, 10e7],
                                   q_L1_du=100)
    dev.set_omega_profile(omega_profile=np.random.permutation(omega_profile_hstack.T).T.ravel(order='F'))
    dev.update_omega_tilde_k(0, deterministic=False)
    grid.add_device(dev)

grid.mpc_controller.set_std_obj_atoms(q_y=cost_profile[:N_tilde])
grid.build_grid()
grid.mpc_controller.build()

print(f"Time to build:{time.time() - st}")

#
st = time.time()
grid.mpc_controller.solve(verbose=True, TimeLimit=200)
print(f"Time to solve including data transfer:{time.time() - st}")

x_k = []
for dev in devices:
    x_k.append(dev.mpc_controller.variables.x.var_N_tilde.value)

x = pd.DataFrame(np.hstack(x_k))

x.plot(drawstyle='steps-post')

u_k = []
for dev in devices:
    u_k.append(dev.mpc_controller.variables.u.var_N_tilde.value)

u = pd.DataFrame(np.hstack(u_k))

u.plot(drawstyle='steps-post')

omega_k = []
for dev in devices:
    omega_k.append(dev.mpc_controller.variables.omega.var_N_tilde.value)

omega = pd.DataFrame(np.hstack(omega_k))

omega.plot(drawstyle='steps-post')
