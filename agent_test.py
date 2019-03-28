from models.agents import MpcAgent
from models.micro_grid_models import DewhModel, GridModel
from models.parameters import dewh_param_struct, grid_param_struct
from controllers.mpc_controller.mpc_controller import *
import numpy as np
from matplotlib import pyplot as plt

dewh_model = DewhModel(param_struct=dewh_param_struct)

num_devices = 100
grid_model = GridModel(num_devices=num_devices, param_struct=grid_param_struct)

a = dict.fromkeys(MldModel._field_names, np.random.rand(3, 3))
a.update(b5=np.ones((3, 1)), d5=np.ones((3, 1)), f5=np.ones((3, 1)))
a.update(A=np.random.rand(3, 3), B2=None, D2=None, F2=None)

agent_model2 = MldSystemModel(MldModel(a, nu_l=1))

mld = dewh_model.get_mld_numeric(dewh_param_struct)

import time

start_init = time.clock()

N_p = 96
N_tilde = N_p + 1
agent = MpcAgent('dewh', None, dewh_model, N_p=N_p)
init_time = time.clock()-start_init
# agent2 = MpcAgent('test', None, agent_model2, N_p=N_p)
# agent_g = MpcAgent('grid', None, grid_model, N_p=N_p)

# mld3 = MldModel(D1=1,D2=3,D3=2,D4=4,d5=6)
# agent3 = MpcAgent(agent_model=AgentModel(mld3), N_p=5)
# agent3._mpc_evo_gen.gen_state_input_evolution_matrices(96)



start_build = time.clock()
agent.mpc_controller.x_k = 45
omega_k_tilde = np.random.random((agent.mpc_controller.omega_tilde_k.shape)) * 2500
omega_k_tilde[2:17] = 1000
omega_k_tilde[20:30] = 100
omega_k_tilde[60:80] = 4500
agent.mpc_controller.omega_tilde_k = omega_k_tilde
build_time = time.clock()-start_build

# for i in range(2):
#     omega_k_tilde = np.random.random((agent.mpc_controller.omega_tilde_k.shape)) * 200*i
#     agent.mpc_controller._custom_constraints.append(
#         agent.mpc_controller.gen_evo_constraints(omega_tilde_k=omega_k_tilde))


q_u = np.random.randint(0, 20, (N_tilde, 1))
q_mu = [1000000, 100000]
# Q_mu = np.eye(2)*200
q_u[40:50] = 10
agent.mpc_controller.set_std_obj_weights(q_u=q_u, q_mu=q_mu)
agent.mpc_controller.build()


fig = plt.figure()
for i in range(1):
    # omega_k_tilde = np.random.random((agent.mpc_controller.omega_tilde_k.shape)) * 1000
    # omega_k_tilde = np.ones(agent.mpc_controller.omega_tilde_k.shape) * 2400
    agent.mpc_controller.omega_tilde_k = omega_k_tilde

    f = agent.mpc_controller.feedback(verbose=True, TimeLimit=5)

    x_k_tilde = agent.mpc_controller.opt_vars.x.var_N_p.value
    u_k_tilde = agent.mpc_controller.opt_vars.u.var_N_p.value
    omega_k_tilde = agent.mpc_controller.omega_tilde_k.value
    cost_u_tilde = q_u

    T_max = agent.agent_model.param_struct.T_h_max
    T_min = agent.agent_model.param_struct.T_h_min
    # plt.close('all')
    # plt.figure()
    ax1 = plt.subplot(4, 1, 1)
    plt.plot(np.arange(len(x_k_tilde)), x_k_tilde, 'k', drawstyle="steps-post", label='x')
    plt.plot([0,len(x_k_tilde)],[T_max, T_max] , 'r--', drawstyle="steps-post",
             label='T_max')
    plt.plot([0,len(x_k_tilde)],[T_min, T_min], 'b--', drawstyle="steps-post",
             label='T_min')
    plt.legend() if i == 0 else None

    plt.subplot(4, 1, 2, sharex=ax1)
    plt.plot(np.arange(len(omega_k_tilde)), omega_k_tilde, 'r', drawstyle="steps-post", label='omega')
    plt.legend() if i == 0 else None

    plt.subplot(4, 1, 3, sharex=ax1)
    plt.plot(np.arange(len(u_k_tilde)), u_k_tilde, 'y', drawstyle="steps-post", label='u')
    plt.legend() if i == 0 else None

    plt.subplot(4, 1, 4, sharex=ax1)
    plt.plot(np.arange(len(cost_u_tilde)), cost_u_tilde, 'g', drawstyle="steps-post", label='price_of_energy')
    plt.legend() if i == 0 else None
    # plt.figure()
    # plt.plot(np.arange(96),agent.mpc_controller.omega_tilde_k.value[:-1], drawstyle="steps-post")

    print(f)
    print(agent.mpc_controller.problem.solver_stats.solve_time)

print(f"Time to init agent:{init_time}")
print(f"Time to build agent:{build_time}")