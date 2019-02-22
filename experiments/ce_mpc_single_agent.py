from models.agents import MpcAgent
from models.micro_grid_agents import DewhModel, GridModel
from models.parameters import dewh_p, grid_p
from controllers.mpc_controller.mpc_controller import *
import numpy as np
from matplotlib import pyplot as plt

dewh_model = DewhModel(param_struct=dewh_p)

liters_p_s = 200/24/60/60*5

sim_steps = 200
N_p = 24
N_tilde = N_p + 1
omega_profile = np.random.random((sim_steps + N_tilde + 100, 1)) * liters_p_s
omega_profile[50:55] = 0.04
cost_profile = np.random.random((sim_steps + N_tilde + 100, 1))


def run_test(N_p=N_p, MIPGap=1e-4):
    N_p = N_p
    N_tilde = N_p + 1
    agent = MpcAgent('dewh', None, dewh_model, N_p=N_p)
    x_0 = 50
    x_k = x_0
    x_out = np.zeros((sim_steps + 1, 1))
    u_in = np.zeros((sim_steps, 1))
    omega_in = np.zeros((sim_steps, 1))
    price = np.zeros((sim_steps, 1))
    q_mu = [10000, 1000]

    Q_mu = np.array([[1000,0],[0,100]])
    agent.mpc_controller.set_std_obj_weights(Q_mu=Q_mu)

    x_out[0, :] = x_0
    for k in range(sim_steps):
        omega_tilde_k = omega_profile[k:N_tilde + k]
        q_u = cost_profile[k:N_tilde + k]
        agent.mpc_controller.update_std_obj_weights(q_u=q_u)

        f = agent.mpc_controller.feedback(x_k=x_k, omega_tilde_k=omega_tilde_k, TimeLimit=2,
                                              MIPGap=MIPGap)

        omega_in[k] = omega_tilde_k[0]
        sim = agent.mpc_controller.mld_numeric_k.lsim_k(x_k=f.x_k, v_k=f.v_k, omega_k=omega_in[k])

        price[k] = q_u[0]
        x_k = x_out[k + 1, :] = sim.x_k1
        u_in[k] = f.u_k

        if k % 10 == 0:
            print(k)

    print(u_in.T @ price)

    fig = plt.figure()

    T_max = agent.agent_model.param_struct.T_h_max
    T_min = agent.agent_model.param_struct.T_h_min
    # plt.close('all')
    # plt.figure()
    ax1 = plt.subplot(4, 1, 1)
    plt.plot(np.arange(len(x_out)), x_out, 'k', drawstyle="steps-post", label='x')
    plt.plot([0, len(x_out)], [T_max, T_max], 'r--', drawstyle="steps-post",
             label='T_max')
    plt.plot([0, len(x_out)], [T_min, T_min], 'b--', drawstyle="steps-post",
             label='T_min')
    plt.legend()

    plt.subplot(4, 1, 2, sharex=ax1)
    plt.plot(np.arange(len(omega_in)), omega_in*dewh_p.dt, 'r', drawstyle="steps-post", label='omega')
    plt.legend()

    plt.subplot(4, 1, 3, sharex=ax1)
    plt.plot(np.arange(len(u_in)), u_in, 'y', drawstyle="steps-post", label='u')
    plt.legend()

    plt.subplot(4, 1, 4, sharex=ax1)
    plt.plot(np.arange(len(price)), price, 'g', drawstyle="steps-post", label='price')
    plt.legend()


run_test()
