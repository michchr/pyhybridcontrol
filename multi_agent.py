from models.micro_grid_agents import GridAgentMpc, DewhAgentMpc, PvAgentMpc, ResDemandAgentMpc
from models.parameters import dewh_param_struct, grid_param_struct

from controllers.mpc_controller import MpcController
from controllers.theromstat_control import DewhTheromstatController
from controllers.no_controller import NoController

import numpy as np
import os
import pandas as pd
from datetime import datetime as DateTime
from tools.tariff_generator import TariffGenerator
import itertools
import cvxpy as cvx

IDX = pd.IndexSlice

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
                                           grid_param_struct.control_dt) / 3600 * grid_param_struct.dt / 100/ 1000

grid = GridAgentMpc(N_p=N_p)
grid.set_price_profile(price_profile=cost_profile)

import time

st = time.time()

devices = [DewhAgentMpc(N_p=N_p) for i in range(10)]

pvAgent = PvAgentMpc(N_p=N_p)
pvAgent.set_omega_profile(omega_pv_profile)

resAgent = ResDemandAgentMpc(N_p=N_p)
resAgent.set_omega_profile(omega_resd_profile)

devices.extend([pvAgent, resAgent])



print(f"Time to create dewh's':{time.time() - st}")

omega_profile_hstack = omega_profile.values.reshape(96, -1, order='F')

np.random.seed(100)


mpc_controllers = ['mpc_pb', 'mpc_ce', 'mpc_scenario']
rule_based_controllers = ['thermo']
controllers = mpc_controllers+rule_based_controllers
deterministic = [True, False, False, False]
deterministic_struct = {cname:det for cname,det in zip(mpc_controllers+rule_based_controllers, deterministic)}




for ind, dev in enumerate(devices):
    if isinstance(dev, DewhAgentMpc):
        dev.set_omega_profile(omega_profile=np.random.permutation(omega_profile_hstack.T).T.ravel(order='F'))
        dev.set_omega_scenarios(omega_scenarios_profile=omega_profile)
    grid.add_device(dev)


for cname in controllers:
    for dev in itertools.chain([grid], devices):
        if cname in mpc_controllers:
            dev.add_controller(cname, MpcController, N_p=N_p)
            if isinstance(dev, DewhAgentMpc):
                dev.set_device_objective_atoms(controller_name=cname, q_mu=[1e5, 1e3],
                                               q_L1_du=0)
        elif cname=='thermo':
            if isinstance(dev, DewhAgentMpc):
                dev.add_controller(cname, DewhTheromstatController, N_p=0, N_tilde=1)

            else:
                dev.add_controller(cname, NoController, N_p=0, N_tilde=1)


for dev in devices:
    if isinstance(dev, DewhAgentMpc):
        dev.x_k = np.random.randint(55, 65)

for k in range(0, 1000):
    st = time.time()
    prices_tilde = grid.get_price_tilde_k(k=k)

    for cname in mpc_controllers:
        for dev in itertools.chain([grid], devices):
            if isinstance(dev, DewhAgentMpc):
                if cname == 'mpc_scenario':
                    omega_tilde_scenarios = dev.get_omega_tilde_scenario(k, N_tilde=N_tilde, num_scenarios=100)
                    dev.controllers[cname].set_constraints(
                        other_constraints=[
                            dev.controllers[cname].gen_evo_constraints(N_tilde=4,
                                                                    omega_scenarios_k=omega_tilde_scenarios)])
            elif isinstance(dev, GridAgentMpc) and cname in mpc_controllers:
                dev.controllers[cname].set_std_obj_atoms(q_L1_y=prices_tilde[cname])

    grid.build_grid(k=k, deterministic_or_struct=deterministic_struct)
    grid.solve_grid_mpc(k=k, verbose=False, TimeLimit=20)
    print(f'k={k}')
    print(f"Time to solve including data transfer:{time.time() - st}\n")
    grid.sim_step_k(k=k)
#
# dfs = [grid.sim_log.get_concat_log()]
# keys = [(grid.device_type, grid.device_id)]
# for dev in grid.devices_struct:
#     dfs.append(dev.sim_log.get_concat_log())
#     keys.append((dev.device_type, dev.device_id))
#
# df = pd.concat(dfs, keys=keys, axis=1)
# IDX = pd.IndexSlice
# df.loc[:, IDX['dewh', :, ('x',)]].plot(drawstyle='steps-post')
