from models.agents import MpcAgent
from models.micro_grid_agents import DewhModel, GridModel
from models.parameters import dewh_p, grid_p
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
with open(file_path,'r') as file_object:
    line_reader = file_object.readlines()
    data = []
    for line in line_reader:
        data.append(float(line.strip())/60.0)

data = np.array(data)
raw_df = pd.DataFrame(data, index=pd.date_range(DateTime(2018,12,1),None, len(data), '1Min'), columns=['actual'])

df:pd.DataFrame = raw_df.resample('15Min').sum()

dewh_control_model = DewhModel(param_struct=dewh_p, const_heat=True)
dewh_sim_model = DewhModel(param_struct=dewh_p, const_heat=False)

sim_steps = 1000
N_p = 24
N_tilde = N_p + 1

omega_profile = df.values/dewh_p.dt
omega_profile_hat = omega_profile[96:]
omega_profile_act = omega_profile[96:]

np.random.seed(100)
tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                                 high_stnd=102.95, high_peak=339.77)

time_0 = DateTime(2018, 12, 1)
cost_profile = tariff_gen.get_price_vector(len(omega_profile), time_0, dewh_p.control_dt)

import time

def run_test(N_p=N_p, sim_steps=sim_steps, MIPGap=1e-4):
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

    price = cost_profile[:sim_steps+1]/100


    agent = MpcAgent(device_type='dewh',
                     device_id=None,
                     sim_model=dewh_sim_model,
                     control_model=dewh_control_model,
                     N_p=N_p)

    q_mu = [10e9, 10e3]
    Q_mu = np.array([[10000,0],[0,10]])
    agent.mpc_controller.set_std_obj_weights(q_mu=q_mu)

    x_out_mpc[0, :] = x_k_mpc
    x_out_therm[0,:] = x_k_therm
    for k in range(sim_steps):
        omega_tilde_k_hat = omega_profile_hat[k:N_tilde + k]
        omega_tilde_k_act = omega_profile_act[k:N_tilde + k]
        q_u = cost_profile[k:N_tilde + k]*dewh_p.P_h_Nom/1000*dewh_p.dt/60
        agent.mpc_controller.update_std_obj_weights(q_u=q_u)

        st = time.clock()
        agent.mpc_controller.build()
        f = agent.feedback(x_k=x_k_mpc, omega_tilde_k=omega_tilde_k_hat, TimeLimit=2,
                                              MIPGap=MIPGap, verbose=False)
        print(time.clock() - st)

        omega_act[k] = omega_tilde_k_act[0]
        omega_hat[k] = omega_tilde_k_hat[0]
        D_h = omega_act[k,0]

        T_h = x_k_mpc if x_k_mpc > dewh_p.T_w else dewh_p.T_w + 0.1
        mld_mpc_sim = agent.sim_model.get_mld_numeric(D_h=D_h, T_h=T_h)
        sim = mld_mpc_sim.lsim_k(x_k=x_k_mpc, u_k=f.u_k, omega_k=D_h)

        x_k_mpc = x_out_mpc[k + 1, :] = sim.x_k1[0, 0]
        u_in_mpc[k] = f.u_k

        #thermo
        T_h_therm = x_k_therm if x_k_therm > dewh_p.T_w else dewh_p.T_w + 0.1
        mld_therm_sim = agent.sim_model.get_mld_numeric(D_h=D_h, T_h=T_h_therm)
        if x_k_therm <= dewh_p.T_h_min:
            u_k_therm=1
        elif x_k_therm >= dewh_p.T_h_max:
            u_k_therm = 0
        elif u_k_therm_kneg1 == 1:
            u_k_therm = 1
        else:
            u_k_therm = 0

        u_k_therm_kneg1 = u_k_therm

        sim_therm = mld_therm_sim.lsim_k(x_k=x_k_therm, u_k=u_k_therm, omega_k=D_h)
        x_k_therm = x_out_therm[k + 1, :] = sim_therm.x_k1[0,0]
        u_in_therm[k] = atleast_2d_col(u_k_therm)

        if k % 10 == 0:
            print(k)


    df = pd.DataFrame(np.hstack([x_out_mpc, x_out_therm, u_in_mpc, u_in_therm, omega_act, omega_hat, price]),
                      index = pd.date_range(pd.datetime(2018,12,1), periods=sim_steps+1, freq='15Min'),
                      columns=['x_out_mpc','x_out_therm','u_in_mpc','u_in_therm','omega_act','omega_hat','price'])

    print("mpc_cost", u_in_mpc.T @ price)
    print("therm_cost", u_in_therm.T @ price)

    return df


def plot_df(df):
    T_max = dewh_p.T_h_max
    T_min = dewh_p.T_h_min


    fig:plt.Figure = plt.figure(tight_layout={'pad':1.0}, figsize=(19.2, 9.28))
    ax1:plt.Axes = fig.add_subplot(5,1,1)
    df['x_out_mpc'].plot(color='k', drawstyle="steps-post", ax=ax1)
    cost_mpc = df['u_in_mpc'].T @ df['price']
    ax1.set_title(f'Temperature Profile - MPC Control (Total Cost = {int(cost_mpc)})')
    ax1.axhline(y=T_max, linestyle='--', color='r', label='T_max')
    ax1.axhline(y=T_min, linestyle='--', color='b', label='T_min')
    ax1.set_ylabel('Temp\n(deg c)', wrap=True)
    plt.legend(loc=1)

    ax2: plt.Axes = fig.add_subplot(5,1,2, sharex=ax1, sharey=ax1)
    df['x_out_therm'].plot(color='k', drawstyle="steps-post", ax=ax2)

    cost_therm = df['u_in_therm'].T @ df['price']
    ax2.set_title(f'Temperature Profile - Thermostatic Control (Total Cost = {int(cost_therm)})')
    ax2.axhline(y=T_max, linestyle='--', color='r', label='T_max')
    ax2.axhline(y=T_min, linestyle='--', color='b', label='T_min')
    ax2.set_ylabel('Temp\n(deg c)', wrap=True)
    plt.legend(loc=1)

    ax3: plt.Axes = fig.add_subplot(5, 1, 3, sharex=ax1)
    (df['omega_hat']*dewh_p.dt).plot(drawstyle="steps-post", ax=ax3)
    (df['omega_act']*dewh_p.dt).plot(drawstyle="steps-post", ax=ax3)
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
