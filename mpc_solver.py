import numpy as np
from datetime import datetime as DateTime

from tools.tariff_generator import TariffGenerator

from mpc_problem import MpcProblem

import pyomo.environ as pe

import cvxpy as cvx


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

    def generate_optimization_model(self, mpc_problem: MpcProblem = None, temp=None):
        if mpc_problem is None:
            mpc_problem = self.mpc_problem

        obj_struct = mpc_problem.mpc_obj_struct  # S_V and S_W ==> J = S_V*V_tilde + S_W*W_tilde
        cons_struct = mpc_problem.mpc_cons_struct  # G_V, G_d G_W and G_x ==> G_V*V_tilde <= G_d + G_W*W_tilde + G_x*x
        decision_var_types = mpc_problem.decision_var_types

        G_V_csr = cons_struct.G_V.tocsr()

        b = cons_struct.G_d.A + cons_struct.G_x@temp

        opt_model = pe.ConcreteModel()
        numcon, numvar = cons_struct.G_V.shape

        V_index_list = [i for i in range(numvar)]
        Con_Index_list = [i for i in range(numcon)]

        opt_model.V_index = pe.Set(initialize=V_index_list, ordered=True)
        opt_model.Con_index = pe.Set(initialize=Con_Index_list, ordered=True)

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

        opt_model.constraint = pe.Constraint(opt_model.Con_index, rule=set_constraint_rule)

        self.optimization_model = opt_model
        # pe.display(opt_model.constraint)
        # opt_model.write("s.lp", "lp", io_options={"symbolic_solver_labels": True})

    def generate_optimization_model_cvx(self, mpc_problem: MpcProblem = None, temp=None):
        if mpc_problem is None:
            mpc_problem = self.mpc_problem

        obj_struct = mpc_problem.mpc_obj_struct  # S_V and S_W ==> J = S_V*V_tilde + S_W*W_tilde
        cons_struct = mpc_problem.mpc_cons_struct  # G_V, G_d G_W and G_x ==> G_V*V_tilde <= G_d + G_W*W_tilde + G_x*x
        decision_var_types = mpc_problem.decision_var_types

        G_V = cons_struct.G_V

        b = cons_struct.G_d.A + cons_struct.G_x@temp

        numcon, numvar = cons_struct.G_V.shape

        V_var_list = []

        for i in range(numvar):
            if decision_var_types[i, 0] == 'b':
                V_var_list.append(cvx.Bool())
            else:
                V_var_list.append(cvx.Variable())

        V_var = cvx.vstack(V_var_list)

        constraints = [G_V*V_var <= b]
        obj = obj_struct.S_V.T*V_var

        prob = cvx.Problem(cvx.Minimize(obj), constraints)

        prob.solve(solver=cvx.GUROBI)
        print(prob.value)

    def solve(self):
        pass
