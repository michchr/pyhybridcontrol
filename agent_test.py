from models.agents import MpcAgent
from models.micro_grid_agents import DewhModel, GridModel
from models.parameters import dewh_p, grid_p

from controllers.mpc_controller.mpc_controller import *

import numpy as np

dewh_model = DewhModel(param_struct=dewh_p)

num_devices = 100
grid_model = GridModel(num_devices=num_devices, param_struct=grid_p)

a = dict.fromkeys(MldModel._field_names, np.random.rand(3, 3))
a.update(b5=np.ones((3, 1)), d5=np.ones((3, 1)), f5=np.ones((3, 1)))
a.update(A=np.random.rand(3, 3), B2=None, D2=None, F2=None)

agent_model2 = MldSystemModel(MldModel(a, nu_l=1))

mld = dewh_model.get_mld_numeric(dewh_p)

N_p = 96
N_tilde = N_p + 1
agent = MpcAgent('dewh', None, dewh_model, N_p=N_p)
agent2 = MpcAgent('test', None, agent_model2, N_p=N_p)
agent_g = MpcAgent('grid', None, grid_model, N_p=N_p)

# mld3 = MldModel(D1=1,D2=3,D3=2,D4=4,d5=6)
# agent3 = MpcAgent(agent_model=AgentModel(mld3), N_p=5)
# agent3._mpc_evo_gen.gen_state_input_evolution_matrices(96)

from matplotlib import pyplot as plt
agent.mpc_controller.x_k = 46
omega_k_tilde = np.random.random((agent.mpc_controller.omega_tilde_k.shape)) * 0
omega_k_tilde[2:17] = 0
omega_k_tilde[60:68] = 0
agent.mpc_controller.omega_tilde_k = omega_k_tilde

for i in range(2):
    omega_k_tilde = np.random.random((agent.mpc_controller.omega_tilde_k.shape)) * 200*i
    agent.mpc_controller._custom_constraints.append(
        agent.mpc_controller.gen_evo_constraints(omega_tilde_k=omega_k_tilde))


q_u = np.random.randint(0, 10, (N_tilde, 1))
# q_u[0:80]=np.random.random_integers(0,2,(80,1))
agent.mpc_controller.set_std_obj_weights(q_u=q_u)
agent.mpc_controller.build()


fig = plt.figure()
for i in range(1):
    # omega_k_tilde = np.random.random((agent.mpc_controller.omega_tilde_k.shape)) * 1000
    # omega_k_tilde = np.ones(agent.mpc_controller.omega_tilde_k.shape) * 2400
    agent.mpc_controller.feedback(verbose=True)
    agent.mpc_controller.omega_tilde_k = omega_k_tilde
    x_k_tilde = agent.mpc_controller.opt_vars.x.var_N_p.value
    u_k_tilde = agent.mpc_controller.opt_vars.u.var_N_p.value
    omega_k_tilde = agent.mpc_controller.omega_tilde_k.value
    cost_u_tilde = q_u

    # plt.close('all')
    # plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(len(x_k_tilde)), x_k_tilde, 'k', drawstyle="steps-post", label='x')
    plt.legend() if i == 0 else None

    plt.subplot(4, 1, 2)
    plt.plot(np.arange(len(omega_k_tilde)), omega_k_tilde, 'r', drawstyle="steps-post", label='omega')
    plt.legend() if i == 0 else None

    plt.subplot(4, 1, 3)
    plt.plot(np.arange(len(u_k_tilde)), u_k_tilde, 'y', drawstyle="steps-post", label='u')
    plt.legend() if i == 0 else None

    plt.subplot(4, 1, 4)
    plt.plot(np.arange(len(cost_u_tilde)), cost_u_tilde, 'g', drawstyle="steps-post", label='cost')
    plt.legend() if i == 0 else None
    # plt.figure()
    # plt.plot(np.arange(96),agent.mpc_controller.omega_tilde_k.value[:-1], drawstyle="steps-post")

print(f)
print(agent.mpc_controller.problem.solver_stats.solve_time)
