import scipy.sparse as scs
import scipy.linalg as scl
import numpy as np
from datetime import datetime as DateTime, timedelta as TimeDelta

from structdict import StructDict

from model_generators import DewhModelGenerator, GridModelGenerator
from dewh_repository import DewhRepository
from micro_grid_model import MicroGridModel

from tariff_generator import TariffGenerator

from mpc_problem import MpcProblem

import pyomo.environ as pe

import inspect

import pprint


class MpcSolver():

    def __init__(self):
        self._mpc_problem = None
        self._optimization_model = None
        self._decision_var_types = None

    @property
    def optimization_model(self):
        return self._optimization_model

    @optimization_model.setter
    def optimization_model(self, optimization_model):
        self._optimization_model = optimization_model

    @property
    def mpc_problem(self):
        return self._mpc_problem

    def load_mpc_problem(self, mpc_problem):
        self._mpc_problem = mpc_problem

    def generate_optimization_model(self, mpc_problem: MpcProblem = None):
        if mpc_problem is None:
            mpc_problem = self.mpc_problem

        obj_struct = mpc_problem.mpc_obj_struct  # S_V and S_W ==> J = S_V*V_tilde + S_W*W_tilde
        cons_struct = mpc_problem.mpc_cons_struct  # G_V, G_d G_W and G_x ==> G_V*V_tilde <= G_d + G_W*W_tilde + G_x*x
        decision_var_types = mpc_problem.decision_var_types

        G_V_csr = cons_struct.G_V.tocsr()

        pprint.pprint(obj_struct.S_V)

        b = cons_struct.G_d.A + cons_struct.G_x*60

        opt_model = pe.ConcreteModel()
        (numcon, numvar) = cons_struct.G_V.shape

        V_index_list = [i for i in range(numvar)]

        opt_model.V_index = pe.Set(initialize=V_index_list, ordered=True)

        ## Set up decision variable ##

        def V_var_set_var_type_rule(opt_model, index):
            if decision_var_types[index, 0] == 'b':
                return pe.Binary
            else:
                return pe.Reals

        opt_model.V_var = pe.Var(opt_model.V_index, domain=V_var_set_var_type_rule, initialize=0)

        ## Set up objective ##

        def set_objective_rule(opt_model):
            expr = sum(obj_struct.S_V[i, 0] * opt_model.V_var[i] for i in V_index_list)
            return expr

        opt_model.objective = pe.Objective(rule=set_objective_rule)

        def set_constraint_rule(opt_model, index):
            row = G_V_csr.getrow(index)
            if row.data.size == 0:
                return pe.Constraint.Skip
            else:
                return sum(coeff * opt_model.V_var[index] for (index, coeff) in zip(row.indices, row.data)) <= b[
                    index, 0]

        opt_model.constraint = pe.Constraint(opt_model.V_index, rule=set_constraint_rule)

        # pe.display(opt_model.constraint)
        opt_model.write("s.lp", "lp", io_options={"symbolic_solver_labels": True})

    def solve(self):
        pass


#

#
#
# A_csr = scs.csr_matrix(A)
#
# con_Index_list = list(range(b.shape[0]))
# model.Con_index = pe.Set(initialize=con_Index_list, ordered=True)
#
#
# def Mod_con(Model,ri):
#     row = A_csr.getrow(ri)
#     if row.data.size == 0:
#         return pe.Constraint.Skip
#     else:
#         return sum(coeff*model.V_var[index] for (index, coeff) in zip(row.indices,row.data)) <= -b[ri,0]
# model.ConF1 = pe.Constraint(model.Con_index, rule=Mod_con)
#
#
# model.write("s.lp", "lp", io_options={"symbolic_solver_labels":True})

################################################################################
################################    MAIN     ###################################
################################################################################

if __name__ == '__main__':
    import timeit
    from parameters import dewh_p, grid_p


    def main():
        N_h = 1
        N_p = 3

        dewh_repo = DewhRepository(DewhModelGenerator)
        dewh_repo.default_param_struct = dewh_p

        for i in range(N_h):
            dewh_repo.add_device_by_default_data(i)

        mg_model = MicroGridModel()
        mg_model.grid_param_struct = grid_p
        mg_model.date_time_0 = DateTime(2018, 5, 1)

        mg_model.add_device_repository(dewh_repo)
        mg_model.gen_concat_device_system_mld()

        mg_model.gen_power_balance_constraint_mld()

        mg_model.get_decision_var_type()

        mpc_prob = MpcProblem(mg_model, N_p=N_p)
        tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                                     high_stnd=102.95, high_peak=339.77)

        mpc_prob.tariff_generator = tariff_gen

        mpc_prob.gen_mpc_objective()

        # mpc_prob.gen_device_state_evolution_matrices()
        # mpc_prob.gen_device_cons_evolution_matrices()
        # mpc_prob.gen_grid_cons_evolution_matrices()

        mpc_prob.gen_mpc_cons_matrices()
        mpc_prob.gen_mpc_objective()
        mpc_prob.gen_mpc_decision_var_types()

        print(mpc_prob.decision_var_types)

        # pprint.pprint(mpc_prob._mg_mpc_obj_struct)
        # pprint.pprint(mpc_prob._mg_mpc_cons_struct)
        #
        # print(mpc_prob._mg_mpc_obj_struct.S_V.T.shape)
        # print(mpc_prob._mg_mpc_obj_struct.S_W.T.shape)
        #
        # print(mpc_prob._mg_mpc_cons_struct.G_V.shape)
        # print(mpc_prob._mg_mpc_cons_struct.G_W.shape)

        mpc_solver = MpcSolver()
        mpc_solver.load_mpc_problem(mpc_prob)
        mpc_solver.generate_optimization_model()


    def func():
        def closure():
            main()
            return 1

        return closure


    t1 = timeit.timeit(func(), number=1)
    print(t1)
