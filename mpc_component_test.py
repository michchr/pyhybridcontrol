from controllers.mpc_controller.mpc_components import  *

mld = MldModel(A=np.ones((2,2)),B1=[1,1],B2=[1,1],B3=[1,1],B4=[1,1],b5=[1,1])

lin_weight = MpcVectorWeight(mld_numeric_k=mld, N_p=10, N_tilde=10, var_name='x')

mat_weight = MpcMatrixWeight(mld_numeric_k=mld, N_p=10, N_tilde=10, var_name='x')