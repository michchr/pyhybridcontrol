from models.agents import MpcAgent
from models.micro_grid_agents import GridAgentMpc, DewhAgentMpc, PvAgentMpc, ResDemandAgentMpc
from models.micro_grid_models import DewhModel, GridModel
from models.parameters import dewh_param_struct, grid_param_struct
from controllers.mpc_controller.mpc_controller import *
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from datetime import datetime as DateTime
from tools.tariff_generator import TariffGenerator

omega_scenarios_profile = pd.read_pickle(
    os.path.realpath(r'./experiments/data/dewh_omega_profile_df.pickle')) / dewh_param_struct.dt

omega_pv_profile = pd.read_pickle(
    os.path.realpath(r'./experiments/data/pv_supply_norm_1000w_15min.pickle'))/1000

omega_resd_profile = pd.read_pickle(
    os.path.realpath(r'./experiments/data/res_demand_norm_1000w_15min.pickle'))/1000

sim_steps = 1000
N_p = 48
N_tilde = N_p + 1

omega_profile = omega_scenarios_profile

np.random.seed(100)
tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                             high_stnd=102.95, high_peak=339.77)

time_0 = DateTime(2018, 12, 1)
cost_profile = tariff_gen.get_price_vector(time_0, len(omega_profile),
                                           dewh_param_struct.control_dt) / 3600 * grid_param_struct.dt / 100 / 1000
grid = GridAgentMpc(N_p=N_p)

import time

st = time.time()

devices = [DewhAgentMpc(N_p=N_p) for i in range(1)]

pvAgent = PvAgentMpc(N_p=N_p)
pvAgent.set_omega_profile(omega_pv_profile)

resAgent = ResDemandAgentMpc(N_p=N_p)
resAgent.set_omega_profile(omega_resd_profile)

devices.extend([pvAgent, resAgent])



print(f"Time to create dewh's':{time.time() - st}")

omega_profile_hstack = omega_profile.values.reshape(96, -1, order='F')

np.random.seed(100)

for ind, dev in enumerate(devices):

    if dev.device_type =='dewh':
        dev.x_k = np.random.randint(55, 65)
        dev.set_device_objective_atoms(q_mu=np.array(np.random.uniform(0.8, 1)) * [1000000, 100000],
                                       q_L1_du=10)
        dev.set_omega_profile(omega_profile=np.random.permutation(omega_profile_hstack.T).T.ravel(order='F'))
        dev.set_omega_scenarios(omega_scenarios_profile=omega_profile)
    dev.update_omega_tilde_k(0, deterministic=False)
    grid.add_device(dev)
    grid.build_grid()

for k in range(0, 10):
    st = time.time()
    for dev in grid.devices:
        dev.update_omega_tilde_k(k, deterministic=False)
        if dev.device_type == 'dewh':
            omega_tilde_scenarios = dev.get_omega_tilde_scenario(k, num_scenarios=10)
            dev.mpc_controller.set_constraints(
                other_constraints=[
                    dev.mpc_controller.gen_evo_constraints(N_tilde=N_tilde, omega_scenarios_k=omega_tilde_scenarios)])

    grid.mpc_controller.set_std_obj_atoms(q_z=cost_profile[k:k + N_tilde])
    grid.build_grid(k=k)
    grid.solve_grid_mpc(verbose=False, TimeLimit=10)
    print(f'k={k}, obj={grid.mpc_controller.problem.value}')
    print(f"Time to solve including data transfer:{time.time() - st}\n"
          f"Solvetime: {grid.mpc_controller.problem.solver_stats.solve_time}")
    grid.sim_step_k(k=k)

dfs = [grid.sim_log.get_concat_log()]
keys = [(grid.device_type, grid.device_id)]
for dev in grid.devices:
    dfs.append(dev.sim_log.get_concat_log())
    keys.append((dev.device_type, dev.device_id))

df = pd.concat(dfs, keys=keys, axis=1)
IDX = pd.IndexSlice
df.loc[:, IDX['dewh', :, ('x',)]].plot(drawstyle='steps-post')
