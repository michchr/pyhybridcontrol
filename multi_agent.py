from models.micro_grid_agents import GridAgentMpc, DewhAgentMpc, PvAgentMpc, ResDemandAgentMpc
from models.parameters import dewh_param_struct, grid_param_struct

from controllers.mpc_controller import MpcController
from controllers.theromstat_control import DewhTheromstatController
from controllers.no_controller import NoController
import time

import numpy as np
import os
import pandas as pd
from datetime import datetime as DateTime
from tools.tariff_generator import TariffGenerator
import itertools

from utils.matrix_utils import atleast_2d_col

from structdict import StructDict
import cvxpy as cvx

IDX = pd.IndexSlice

st = time.time()

OMEGA_DHW_STOCHASTIC_SCENARIOS_PATH = os.path.realpath(
    r'./experiments/data/dewh_dhw_demand_stochastic_scenarios_15Min_200Lpd_mean.pickle')
OMEGA_DHW_ACTUAL_SCENARIOS_PATH = os.path.realpath(
    r'./experiments/data/dewh_dhw_demand_actual_scenarios_15Min_200Lpd_mean.pickle')
OMEGA_PV_PROFILE_PATH = os.path.realpath(
    r'./experiments/data/pv_supply_norm_1w_max_15min_from_091218_150219.pickle')
OMEGA_RESD_PROFILE_PATH = os.path.realpath(
    r'./experiments/data/res_demand_norm_1w_mean_15min_from_091218_150219.pickle')

omega_dhw_stochastic_scenarios = np.divide(pd.read_pickle(OMEGA_DHW_STOCHASTIC_SCENARIOS_PATH),
                                           dewh_param_struct.ts)  # in L/s
omega_dhw_actual_scenarios = np.divide(pd.read_pickle(OMEGA_DHW_ACTUAL_SCENARIOS_PATH),
                                       dewh_param_struct.ts)  # in L/s
omega_pv_profile = pd.read_pickle(OMEGA_PV_PROFILE_PATH)  # in W with P_pv_max=1
omega_resd_profile = pd.read_pickle(OMEGA_RESD_PROFILE_PATH)  # in W with P_resd_ave=1

print(f"Time to load historical data and scenarios':{time.time() - st}")

steps_per_day = int(pd.Timedelta('1D').total_seconds() / dewh_param_struct.ts)
N_p_max = 96
N_tilde_max = N_p_max + 1

N_h = 1
time_0 = DateTime(2018, 12, 10)

N_p = 48
N_tilde = N_p + 1

sim_steps = 1
max_steps = int(sim_steps + 2 * N_tilde_max)


def get_actual_omega_dewh_profiles(actual_scenarios=None, N_h=1, size=1):
    _, num_scen = actual_scenarios.shape
    omega_dewh_profiles = StructDict()
    for i in range(1, N_h + 1):
        randstate = np.random.RandomState(seed=np.int32(i ** 2))
        profile = actual_scenarios[:, randstate.choice(num_scen, size=size, replace=False)]
        omega_dewh_profiles[i] = profile.reshape(-1, 1, order='F')
    return omega_dewh_profiles


def get_dewh_random_initial_state(dev_id):
    randstate = np.random.RandomState(seed=np.int32(dev_id ** 2))
    return randstate.randint(55, 65)


def get_min_max_dhw_scenario(k, N_tilde, min_or_max):
    min_or_max = min_or_max.flatten()
    if len(min_or_max) != steps_per_day:
        raise ValueError("Invalid shape for min_or_max")
    pos = k % steps_per_day
    mult = N_tilde // steps_per_day + 1
    return atleast_2d_col(np.roll(np.tile(min_or_max, mult), -pos)[:N_tilde])


omega_dewh_profiles_struct = get_actual_omega_dewh_profiles(actual_scenarios=omega_dhw_actual_scenarios.values,
                                                            N_h=N_h, size=max_steps)

tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                             high_stnd=102.95, high_peak=339.77)

cost_profile = tariff_gen.get_price_vector(time_0, max_steps,
                                           grid_param_struct.control_ts) / 3600 / 100 / 1000 * grid_param_struct.ts

# todo Still needs work
DewhAgentMpc.delete_all_devices()
GridAgentMpc.delete_all_devices()

grid = GridAgentMpc(device_id=1)
grid.set_price_profile(price_profile=cost_profile)

st = time.time()

devices = [DewhAgentMpc(device_id=i) for i in range(1, N_h + 1)]

pvAgent = PvAgentMpc()
pvAgent.set_omega_profile(omega_pv_profile)

resAgent = ResDemandAgentMpc()
resAgent.set_omega_profile(omega_resd_profile)

devices.extend([pvAgent, resAgent])

print(f"Time to create dewh's':{time.time() - st}")

mpc_controllers = ['mpc_pb', 'mpc_ce', 'mpc_scenario', 'mpc_minmax']
rule_based_controllers = ['thermo']
controllers = mpc_controllers + rule_based_controllers
deterministic = [True, False, False, False, False]
deterministic_struct = {cname: det for cname, det in zip(mpc_controllers + rule_based_controllers, deterministic)}

min_dhw_day = omega_dhw_stochastic_scenarios.min(axis=1).values
max_dhw_day = omega_dhw_stochastic_scenarios.max(axis=1).values
omega_dhw_stochastic_scenarios_profile = omega_dhw_stochastic_scenarios.values.flatten(order='f')
for ind, dev in enumerate(devices):
    if isinstance(dev, DewhAgentMpc):
        dev.set_omega_profile(omega_profile=omega_dewh_profiles_struct[dev.device_id])
        dev.set_omega_scenarios(omega_scenarios_profile=omega_dhw_stochastic_scenarios_profile)
    grid.add_device(dev)

df_sim = StructDict()
for N_p in [48]:
    N_tilde=N_p+1
    for dev in itertools.chain([grid], devices):
        dev.delete_all_controllers()

    for cname in controllers:
        for dev in itertools.chain([grid], devices):
            if cname in mpc_controllers:
                dev.add_controller(cname, MpcController, N_p=N_p)
                if isinstance(dev, DewhAgentMpc):
                    dev.set_device_objective_atoms(controller_name=cname, q_mu=[1e3, 5],
                                                   q_L1_du=0)
            elif cname == 'thermo':
                if isinstance(dev, DewhAgentMpc):
                    dev.add_controller(cname, DewhTheromstatController, N_p=0, N_tilde=1)

                else:
                    dev.add_controller(cname, NoController, N_p=0, N_tilde=1)

    for dev in devices:
        if isinstance(dev, DewhAgentMpc):
            dev.x_k = get_dewh_random_initial_state(dev.device_id)

    prices = StructDict({cname: 0 for cname in controllers})

    grid.build_grid(k=0, deterministic_or_struct=deterministic_struct)
    for k in range(0, sim_steps):
        st = time.time()
        prices_tilde = grid.get_price_tilde_k(k=k)

        for cname in mpc_controllers:
            for dev in itertools.chain([grid], devices):
                if isinstance(dev, DewhAgentMpc):
                    if cname.startswith('mpc'):
                        price_vec = prices_tilde[cname]
                        max_cost = np.sum(price_vec) * dewh_param_struct.P_h_Nom
                        q_mu_top = max_cost*10
                        q_mu_bot = max_cost
                        dev.set_device_objective_atoms(controller_name=cname,
                                                       q_mu=np.hstack([q_mu_top, q_mu_bot]).ravel(order='c'),
                                                       q_L1_du=0)

                    if cname == 'mpc_scenario' or cname == 'mpc_minmax':
                        omega_tilde_scenarios = dev.get_omega_tilde_scenario(k, N_tilde=N_tilde, num_scenarios=20)
                        if cname == 'mpc_scenario':
                            dev.controllers[cname].set_constraints(
                                other_constraints=[
                                    dev.controllers[cname].gen_evo_constraints(
                                        N_tilde=6,
                                        omega_scenarios_k=omega_tilde_scenarios)])

                        if cname == 'mpc_minmax':
                            omega_min = get_min_max_dhw_scenario(k=k, N_tilde=N_tilde, min_or_max=min_dhw_day)
                            omega_max = get_min_max_dhw_scenario(k=k, N_tilde=N_tilde, min_or_max=max_dhw_day)

                            min_cons = dev.controllers[cname].gen_evo_constraints(N_tilde=N_tilde,
                                                                                  omega_tilde_k=omega_min)

                            max_cons = dev.controllers[cname].gen_evo_constraints(N_tilde=N_tilde,
                                                                                  omega_tilde_k=omega_max)

                            dev.controllers[cname].set_constraints(
                                other_constraints=[min_cons, max_cons])
                elif isinstance(dev, GridAgentMpc) and cname in mpc_controllers:
                    dev.controllers[cname].set_std_obj_atoms(q_z=prices_tilde[cname])

        grid.build_grid(k=k, deterministic_or_struct=deterministic_struct)
        grid.solve_grid_mpc(k=k, verbose=False, TimeLimit=10, MIPGap=1e-4)
        print(f'k={k}')
        print(f"Time to solve including data transfer:{time.time() - st}")
        l_sim = grid.sim_step_k(k=k)
        print(f"Total Looptime:{time.time() - st}")

        for cname in controllers:
            prices[cname] += grid.sim_logs[cname].get(k).cost
        print(prices)
        print('\n')

    df_sim[N_p] = grid.grid_sim_dataframe
