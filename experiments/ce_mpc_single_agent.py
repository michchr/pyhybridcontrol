from models.agents import MpcAgent
from models.micro_grid_models import DewhModel
from models.parameters import dewh_param_struct
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from datetime import datetime as DateTime
from tools.tariff_generator import TariffGenerator
from utils.matrix_utils import atleast_2d_col

dewh_control_model = DewhModel(param_struct=dewh_param_struct, const_heat=True)
dewh_sim_model = DewhModel(param_struct=dewh_param_struct, const_heat=False)

sim_steps = 500
N_p = 24
N_tilde = N_p + 1

dt = dewh_sim_model.mld_numeric.mld_info.dt

omega_profile = pd.read_pickle(os.path.realpath(r'./data/dewh_omega_profile_df.pickle')) / dt #liters/s

np.random.seed(100)
tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                             high_stnd=102.95, high_peak=339.77)

time_0 = DateTime(2018, 12, 3)
cost_profile = tariff_gen.get_price_vector(time_0, sim_steps+10000, dewh_param_struct.control_dt)

import time


def run_test(N_p=N_p, sim_steps=sim_steps, MIPGap=1e-4, q_L1_du=1, solver=cvx.GUROBI):
    N_p = N_p
    N_tilde = N_p + 1
    x_0 = 52
    x_k_mpc = x_0
    x_k_therm = x_0
    u_k_therm_kneg1 = 0

    x_out_mpc = np.zeros((sim_steps + 1, 1))
    x_out_therm = np.zeros((sim_steps + 1, 1))
    u_in_mpc = np.zeros((sim_steps + 1, 1))
    u_in_therm = np.zeros((sim_steps + 1, 1))
    omega_act = np.zeros((sim_steps + 1, 1))
    omega_hat = np.zeros((sim_steps + 1, 1))

    price = cost_profile[:sim_steps + 1] / 100

    agent = MpcAgent(device_type='dewh',
                     device_id=None,
                     sim_model=dewh_sim_model,
                     control_model=dewh_control_model,
                     N_p=N_p)

    q_mu = [10e6, 10e5]
    Q_mu = np.array([[10000, 0], [0, 10]])
    agent.mpc_controller.set_std_obj_atoms(q_mu=q_mu)

    x_out_mpc[0, :] = x_k_mpc
    x_out_therm[0, :] = x_k_therm
    for k in range(sim_steps):
        omega_tilde_k_hat = omega_profile_hat[k:N_tilde + k]
        omega_tilde_k_act = omega_profile_act[k:N_tilde + k]
        q_u = cost_profile[k:N_tilde + k] * dewh_param_struct.P_h_Nom / 1000 * dewh_param_struct.dt / 60
        agent.mpc_controller.update_std_obj_atoms(q_u=q_u, q_L1_du=q_L1_du)

        st = time.clock()
        agent.mpc_controller.build()
        try:
            if solver is cvx.GUROBI:
                fb = agent.mpc_controller.feedback(x_k=x_k_mpc, omega_tilde_k=omega_tilde_k_hat, TimeLimit=2,
                                                   MIPGap=MIPGap, verbose=False, solver=solver)
            elif solver is cvx.CPLEX:
                fb = agent.mpc_controller.feedback(x_k=x_k_mpc, omega_tilde_k=omega_tilde_k_hat, verbose=False,
                                                   solver=solver)
        except MpcSolverError:
            print('solver_error')
            agent.mpc_controller.feedback(x_k=x_k_mpc, omega_tilde_k=omega_tilde_k_hat, verbose=False,
                                          solver=cvx.CPLEX,
                                          cplex_filename='test.lp')
            return agent

        print(time.clock() - st, agent.mpc_controller.problem.objective.value)

        omega_act[k] = omega_tilde_k_act[0]
        omega_hat[k] = omega_tilde_k_hat[0]
        D_h = omega_act[k, 0]

        T_h = x_k_mpc if x_k_mpc > dewh_param_struct.T_w else dewh_param_struct.T_w + 0.1
        mld_mpc_sim = agent.sim_model.get_mld_numeric(D_h=D_h, T_h=T_h)
        sim = mld_mpc_sim.lsim_k(x_k=x_k_mpc, u_k=fb.u, omega_k=D_h)

        x_k_mpc = x_out_mpc[k + 1, :] = sim.x_k1[0, 0]
        u_in_mpc[k] = fb.u

        # thermo
        T_h_therm = x_k_therm if x_k_therm > dewh_param_struct.T_w else dewh_param_struct.T_w + 0.1
        mld_therm_sim = agent.sim_model.get_mld_numeric(D_h=D_h, T_h=T_h_therm)
        if x_k_therm <= dewh_param_struct.T_h_min:
            u_k_therm = 1
        elif x_k_therm >= dewh_param_struct.T_h_max:
            u_k_therm = 0
        elif u_k_therm_kneg1 == 1:
            u_k_therm = 1
        else:
            u_k_therm = 0

        u_k_therm_kneg1 = u_k_therm

        sim_therm = mld_therm_sim.lsim_k(x_k=x_k_therm, u_k=u_k_therm, omega_k=D_h)
        x_k_therm = x_out_therm[k + 1, :] = sim_therm.x_k1[0, 0]
        u_in_therm[k] = atleast_2d_col(u_k_therm)

        if k % 10 == 0:
            print(k)

    df = pd.DataFrame(np.hstack([x_out_mpc, x_out_therm, u_in_mpc, u_in_therm, omega_act, omega_hat, price]),
                      index=pd.date_range(pd.datetime(2018, 12, 3), periods=sim_steps + 1, freq='15Min'),
                      columns=['x_out_mpc', 'x_out_therm', 'u_in_mpc', 'u_in_therm', 'omega_act', 'omega_hat', 'price'])

    print("mpc_cost", u_in_mpc.T @ price)
    print("therm_cost", u_in_therm.T @ price)

    return df


def plot_df(df):
    T_max = dewh_param_struct.T_h_max
    T_min = dewh_param_struct.T_h_min

    fig: plt.Figure = plt.figure(tight_layout={'pad': 1.0}, figsize=(19.2, 9.28))
    ax1: plt.Axes = fig.add_subplot(5, 1, 1)
    df['x_out_mpc'].plot(color='k', drawstyle="steps-post", ax=ax1)
    cost_mpc = df['u_in_mpc'].T @ df['price']
    ax1.set_title(f'Temperature Profile - MPC Control (Total Cost = {int(cost_mpc)})')
    ax1.axhline(y=T_max, linestyle='--', color='r', label='T_max')
    ax1.axhline(y=T_min, linestyle='--', color='b', label='T_min')
    ax1.set_ylabel('Temp\n(deg c)', wrap=True)
    plt.legend(loc=1)

    ax2: plt.Axes = fig.add_subplot(5, 1, 2, sharex=ax1, sharey=ax1)
    df['x_out_therm'].plot(color='k', drawstyle="steps-post", ax=ax2)

    cost_therm = df['u_in_therm'].T @ df['price']
    ax2.set_title(f'Temperature Profile - Thermostatic Control (Total Cost = {int(cost_therm)})')
    ax2.axhline(y=T_max, linestyle='--', color='r', label='T_max')
    ax2.axhline(y=T_min, linestyle='--', color='b', label='T_min')
    ax2.set_ylabel('Temp\n(deg c)', wrap=True)
    plt.legend(loc=1)

    ax3: plt.Axes = fig.add_subplot(5, 1, 3, sharex=ax1)
    (df['omega_hat'] * dewh_param_struct.dt).plot(drawstyle="steps-post", ax=ax3)
    (df['omega_act'] * dewh_param_struct.dt).plot(drawstyle="steps-post", ax=ax3)
    ax3.set_title('Hot water demand (Actual and Forecast)')
    ax3.set_ylabel('Hot water demand\n(litres)', wrap=True)
    plt.legend(loc=1)

    ax4: plt.Axes = fig.add_subplot(5, 1, 4, sharex=ax1)
    df['u_in_mpc'].plot(drawstyle="steps-post", ax=ax4)
    df['u_in_therm'].plot(drawstyle="steps-post", ax=ax4)
    ax4.set_title('Control Input (MPC vs Thermostatic Control)')
    ax4.set_ylabel('Input\n(on/off)', wrap=True)
    plt.legend(loc=1)

    ax5: plt.Axes = fig.add_subplot(5, 1, 5, sharex=ax1)
    df['price'].plot(drawstyle="steps-post", ax=ax5)
    ax5.set_title('Energy Price')
    ax5.set_ylabel('Energy price', wrap=True)
    ax5.set_xlabel('Date-time')
    plt.legend(loc=1)

    return fig
