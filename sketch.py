# from models.mld_model import MldModel, MldSystemModel
# from controllers.mpc_controller import MpcController
# import cvxpy as cvx
# from models.agents import MpcAgent
#
#
# mld = MldModel(A=[[0.5, 0],[1,1]], B1=[1,0], b5=[10,0],F1=[1,-1],f5=[0,11],dt=1)
# model = MldSystemModel(mld_numeric=mld)
# mpc_agent = MpcAgent(sim_model=model, N_p=30)
#
# mpc_agent.mpc_controller.set_std_obj_atoms(Q_x_N_p=[[100,0],[0,5]], Q_du=1, Q_u=1000)
# mpc_agent.mpc_controller.sim_log.update(-1,u=0)
# mpc_agent.mpc_controller.build()
#
# print(mpc_agent.mpc_controller.variables_k_neg1)
# mpc_agent.x_k=[10,0]
# for k in range(100):
#     mpc_agent.mpc_controller.solve(k, solver=cvx.GUROBI)
#     # print(mpc_agent.mpc_controller.std_obj_atoms.u.Linear_vector_d.variable.var_N_tilde.value[-1])
#     mpc_agent.mpc_controller.sim_step_k(k)
#     mpc_agent.mpc_controller.sim_log.update(k,cost=mpc_agent.mpc_controller.problem.objective.value)
#
#
# df = mpc_agent.mpc_controller.sim_log.get_concat_log()
# print(df)
#
# df.x.loc[:,0:0].plot(drawstyle='steps-post')





from models.parameters import dewh_param_struct
import os
import pandas as pd

IDX = pd.IndexSlice

omega_scenarios_profile = pd.read_pickle(
    os.path.realpath(r'./experiments/data/dewh_omega_profile_df.pickle')) / dewh_param_struct.dt