from examples.residential_mg_with_pv_and_dewhs.modelling.micro_grid_agents import (GridAgentMpc, DewhAgentMpc,
                                                                                   PvAgentMpc, ResDemandAgentMpc)
from examples.residential_mg_with_pv_and_dewhs.modelling.parameters import (dewh_param_struct, grid_param_struct,
                                                                            res_demand_param_struct, pv_param_struct)
from collections import namedtuple
from controllers.mpc_controller import MpcController
from examples.residential_mg_with_pv_and_dewhs.theromstat_control import DewhTheromstatController
from controllers.no_controller import NoController
import time

import numpy as np
import os
import pandas as pd
from datetime import datetime as DateTime
from examples.residential_mg_with_pv_and_dewhs.tariff_generator import TariffGenerator
import itertools

from utils.matrix_utils import atleast_2d_col

from structdict import StructDict

IDX = pd.IndexSlice

st = time.time()

BASE_FILE = os.path.dirname(__file__)

OMEGA_DHW_STOCHASTIC_SCENARIOS_PATH = os.path.realpath(
    fr'{BASE_FILE}/data/dewh_dhw_demand_stochastic_scenarios_15Min_200Lpd_mean.pickle')
OMEGA_DHW_ACTUAL_SCENARIOS_PATH = os.path.realpath(
    fr'{BASE_FILE}/data/dewh_dhw_demand_actual_scenarios_15Min_200Lpd_mean.pickle')
OMEGA_PV_PROFILE_PATH = os.path.realpath(
    fr'{BASE_FILE}/data/pv_supply_norm_1w_max_15min_from_091218_150219.pickle')
OMEGA_RESD_PROFILE_PATH = os.path.realpath(
    fr'{BASE_FILE}/data/res_demand_norm_1w_mean_15min_from_091218_150219.pickle')

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
sim_steps_max = 2500
max_steps = int(sim_steps_max + 2 * N_tilde_max)

N_h = 20
time_0 = DateTime(2018, 12, 10)


def get_actual_omega_dewh_profiles(actual_scenarios=None, N_h=1, size=1):
    if isinstance(actual_scenarios, pd.DataFrame):
        actual_scenarios = actual_scenarios.values
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


def get_min_max_dhw_scenario(k, N_tilde, min_dhw_day, max_dhw_day):
    min_dhw_day = min_dhw_day.flatten()
    max_dhw_day = max_dhw_day.flatten()
    if len(min_dhw_day) != steps_per_day:
        raise ValueError("Invalid shape for min_dhw_day")
    if len(max_dhw_day) != steps_per_day:
        raise ValueError("Invalid shape for max_dhw_day")
    pos = k % steps_per_day
    mult = N_tilde // steps_per_day + 1

    return [atleast_2d_col(np.roll(np.tile(dwh_day, mult), -pos)[:N_tilde]) for dwh_day in [min_dhw_day, max_dhw_day]]


tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                             high_stnd=102.95, high_peak=339.77)

cost_profile = tariff_gen.get_price_vector(time_0, max_steps,
                                           grid_param_struct.control_ts) / 3600 / 100 / 1000 * grid_param_struct.ts

# todo Still needs work
DewhAgentMpc.delete_all_devices()
GridAgentMpc.delete_all_devices()
PvAgentMpc.delete_all_devices()
ResDemandAgentMpc.delete_all_devices()

st = time.time()

## create grid ##

grid_param_struct_adjusted = grid_param_struct.deepcopy()
grid_param_struct_adjusted.P_g_min = -1e4 * N_h
grid_param_struct_adjusted.P_g_max = 1e4 * N_h
grid = GridAgentMpc(device_id=1, param_struct=grid_param_struct_adjusted)
grid.set_price_profile(price_profile=cost_profile)

## Create devices ##
min_dhw_day = omega_dhw_stochastic_scenarios.min(axis=1).values
max_dhw_day = omega_dhw_stochastic_scenarios.max(axis=1).values
omega_dhw_stochastic_scenarios_profile = omega_dhw_stochastic_scenarios.values.flatten(order='f')
dewh_param_struct_adjusted = dewh_param_struct.deepcopy()
dewh_param_struct_adjusted.T_h_min = 50.0  # C
dewh_param_struct_adjusted.T_h_max = 80.0
dewh_list = [DewhAgentMpc(device_id=i, param_struct=dewh_param_struct_adjusted) for i in range(1, N_h + 1)]

omega_dewh_profiles_struct = get_actual_omega_dewh_profiles(actual_scenarios=omega_dhw_actual_scenarios.values,
                                                            N_h=N_h, size=max_steps)
for dewh in dewh_list:
    dewh.set_omega_profile(omega_profile=omega_dewh_profiles_struct[dewh.device_id])
    dewh.set_omega_scenarios(omega_scenarios_profile=omega_dhw_stochastic_scenarios_profile)
    grid.add_device(dewh)

pv_param_struct_adjusted = pv_param_struct.deepcopy()
pv_param_struct_adjusted.P_pv_units = N_h
pvAgent = PvAgentMpc(device_id=1, param_struct=pv_param_struct_adjusted)
pvAgent.set_omega_profile(omega_pv_profile)
grid.add_device(pvAgent)

res_demand_param_struct_adjusted = res_demand_param_struct.deepcopy()
res_demand_param_struct_adjusted.P_res_units = N_h
resdAgent = ResDemandAgentMpc(device_id=1, param_struct=res_demand_param_struct_adjusted)
resdAgent.set_omega_profile(omega_resd_profile)
grid.add_device(resdAgent)

################################

print(f"Time to create dewh's':{time.time() - st}")

ControllerClass = namedtuple('ControllerClass', [
    'controller_type',
    'is_deterministic'
])
controllers_choices = StructDict(
    {'mpc_pb'        : ControllerClass(MpcController, True),
     'mpc_ce'        : ControllerClass(MpcController, False),
     'mpc_sb_reduced': ControllerClass(MpcController, False),
     'mpc_sb_full'   : ControllerClass(MpcController, False),
     'mpc_minmax'    : ControllerClass(MpcController, False),
     'thermo'        : ControllerClass(DewhTheromstatController, False)
     }
)


def sim_mpc(N_p=1, sim_steps=1, soft_top_mult=10.0, soft_bot_mult=1.0, num_scenarios=20, N_sb_reduced=8,
            controllers=None, save_text_postfix=""):
    N_tilde = N_p + 1
    controllers = controllers or {}
    deterministic_struct = {cname: controller.is_deterministic for cname, controller in controllers.items()}

    for dev in itertools.chain([grid], grid.devices):
        dev.delete_all_controllers()

    for cname, controller in controllers.items():
        for dev in itertools.chain([grid], grid.devices):
            if isinstance(dev, DewhAgentMpc):
                if issubclass(controller.controller_type, MpcController):
                    dev.add_controller(cname, controller.controller_type, N_p=N_p)
                else:
                    dev.add_controller(cname, controller.controller_type, N_p=0, N_tilde=1)
            else:
                if issubclass(controller.controller_type, MpcController):
                    dev.add_controller(cname, controller.controller_type, N_p=N_p)
                else:
                    dev.add_controller(cname, NoController, N_p=0, N_tilde=1)

    for dev in grid.devices:
        if isinstance(dev, DewhAgentMpc):
            dev.x_k = get_dewh_random_initial_state(dev.device_id)

    total_cost_struct = StructDict({cname: 0 for cname in controllers})

    grid.build_grid(k=0, deterministic_or_struct=deterministic_struct)
    for k in range(0, sim_steps):
        st = time.time()
        prices_tilde = grid.get_price_tilde_k(k=k)

        for cname, controller in controllers.items():
            if issubclass(controller.controller_type, MpcController):
                for dev in itertools.chain([grid], grid.devices):
                    if isinstance(dev, DewhAgentMpc) and issubclass(controller.controller_type, MpcController):
                        if cname.startswith('mpc'):
                            price_vec = prices_tilde[cname]
                            max_cost = np.sum(price_vec) * dewh_param_struct.P_h_Nom
                            q_mu_top = max_cost * soft_top_mult
                            q_mu_bot = max_cost * soft_bot_mult
                            dev.set_device_objective_atoms(controller_name=cname,
                                                           q_mu=np.hstack([q_mu_top, q_mu_bot]).ravel(order='c'))

                        if cname.startswith('mpc_sb'):
                            omega_tilde_scenarios = dev.get_omega_tilde_scenario(k, N_tilde=N_tilde,
                                                                                 num_scenarios=num_scenarios)
                            if cname == 'mpc_sb_reduced':
                                dev.controllers[cname].set_constraints(
                                    other_constraints=[
                                        dev.controllers[cname].gen_evo_constraints(
                                            N_tilde=N_sb_reduced,
                                            omega_scenarios_k=omega_tilde_scenarios)])
                            elif cname == 'mpc_sb_full':
                                dev.controllers[cname].set_constraints(
                                    other_constraints=[
                                        dev.controllers[cname].gen_evo_constraints(
                                            omega_scenarios_k=omega_tilde_scenarios)])

                        elif cname == 'mpc_minmax':
                            omega_min, omega_max = get_min_max_dhw_scenario(k=k, N_tilde=N_tilde,
                                                                            min_dhw_day=min_dhw_day,
                                                                            max_dhw_day=max_dhw_day)

                            min_cons = dev.controllers[cname].gen_evo_constraints(N_tilde=N_tilde,
                                                                                  omega_tilde_k=omega_min)

                            max_cons = dev.controllers[cname].gen_evo_constraints(N_tilde=N_tilde,
                                                                                  omega_tilde_k=omega_max)

                            dev.controllers[cname].set_constraints(
                                other_constraints=[min_cons, max_cons])
                    elif isinstance(dev, GridAgentMpc) and issubclass(controller.controller_type, MpcController):
                        dev.controllers[cname].set_std_obj_atoms(q_z=prices_tilde[cname])

        grid.build_grid(k=k, deterministic_or_struct=deterministic_struct)
        grid.solve_grid_mpc(k=k, verbose=False, TimeLimit=20, MIPGap=1e-2)
        print(f'k={k}, N_p={N_p}')
        print(f"Time to solve including data transfer:{time.time() - st}")
        l_sim = grid.sim_step_k(k=k)
        print(f"Total Looptime:{time.time() - st}")

        solve_times_struct = StructDict()
        for cname in controllers:
            total_cost_struct[cname] += grid.sim_logs[cname].get(k).cost
            solve_times_struct[cname] = (grid.sim_logs[cname].get(k).time_in_solver,
                                         grid.sim_logs[cname].get(k).time_solve_overall)
        print('Total_cost\n', total_cost_struct)
        print('Solve_times\n', solve_times_struct)
        print('\n')

    df_sim: pd.DataFrame = grid.grid_sim_dataframe
    df_sim.index = pd.date_range(start=time_0, periods=sim_steps, freq='15min')

    if save_text_postfix and not save_text_postfix.startswith('_'):
        save_text_postfix = '_' + save_text_postfix

    T_max = int(dewh_param_struct_adjusted.T_h_max)
    T_min = int(dewh_param_struct_adjusted.T_h_min)

    save_dir = fr'{BASE_FILE}/sim_out'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.realpath(
        fr'{BASE_FILE}/sim_out/sim_Np_{N_p}_st_{int(soft_top_mult)}_sb_{int(soft_bot_mult)}_Ns_{num_scenarios}_'
        fr'Nsr_{N_sb_reduced}_Nh_{N_h}_Tmax_{T_max}_Tmin_{T_min}{save_text_postfix}.sim_out')

    df_sim.to_pickle(save_path)
    return StructDict(df_sim=df_sim, locals_vars=locals())


controller_names = ['mpc_pb', 'mpc_ce', 'mpc_sb_reduced', 'mpc_sb_full', 'mpc_minmax', 'thermo']
controllers = controllers_choices.get_sub_struct(controller_names)

omega_dewh_profiles_struct = get_actual_omega_dewh_profiles(actual_scenarios=omega_dhw_actual_scenarios.values,
                                                            N_h=50, size=max_steps)
#
# sim_mpc(N_p=24, sim_steps=5, controllers=controllers, num_scenarios=20, N_sb_reduced=6,
#         save_text_postfix=f'')
