from controllers.controller_base import NonPredictiveController, build_required_decor

from controllers.components.variables import VariablesStruct_k, VariablesStruct_k_neg1

import numpy as np

from utils.matrix_utils import atleast_2d_col


class DewhTheromstatController(NonPredictiveController):
    _repr_components = ['mld_numeric_k', 'mld_info_k']

    def reset_components(self, x_k=None, omega_tilde_k=None):
        super(DewhTheromstatController, self).reset_components(x_k=x_k, omega_tilde_k=omega_tilde_k)
        mld_info_k = self.mld_info_k
        self._variables_k = VariablesStruct_k(
            self.sim_model.mld_numeric.lsim_k(x_k=self._x_k, u_k=None, mu_k=None,
                                              omega_k=self._omega_tilde_k[:mld_info_k.nomega]))
        self._variables_k_neg1 = self.get_sim_k(-1)

    @property
    def variables_k(self):
        return self._variables_k

    @property
    def variables_k_neg1(self):
        return self._variables_k_neg1

    @variables_k_neg1.setter
    def variables_k_neg1(self, variables_k_neg1_struct):
        self._variables_k_neg1.update(variables_k_neg1_struct)

    @build_required_decor(set=False)
    def build(self, *args, **kwargs):
        super(DewhTheromstatController, self).build(*args, **kwargs)

    def solve(self, k, x_k=None, omega_tilde_k=None, external_solve=None, solver=None, *args, **kwargs):
        if x_k is not None:
            self.x_k = x_k
        if omega_tilde_k is not None:
            self.omega_tilde_k = omega_tilde_k

        k_neg1 = k - 1 if k is not None else k
        vars_k_neg1 = self.variables_k_neg1 = self.get_sim_k(k=k_neg1)
        u_k_neg1 = np.asscalar(vars_k_neg1.u)
        T_h = np.asscalar(self.x_k)
        T_h_max = self.sim_model.param_struct.T_h_max

        if T_h <= T_h_max - np.random.uniform(low=7, high=11):
            u_k = 1
        elif T_h >= T_h_max - 3:
            u_k = 0
        elif u_k_neg1 == 1:
            u_k = 1
        else:
            u_k = 0

        mld_info = self.mld_info_k
        omega_k = self.omega_tilde_k[:mld_info.nomega]
        self._variables_k = self.sim_step_k(k=k, x_k=T_h, omega_k=omega_k, u_k=u_k, solver=solver, step_state=False)

    def feedback(self, k, x_k=None, omega_tilde_k=None, external_solve=None, solver=None, *args, **kwargs):
        self.solve(k=k, x_k=x_k, omega_tilde_k=omega_tilde_k, external_solve=external_solve,
                   solver=solver, *args, **kwargs)

        return self.variables_k

    @property
    def x_k(self):
        return self._x_k

    @x_k.setter
    def x_k(self, value):
        self._x_k = atleast_2d_col(value)

    @property
    def omega_tilde_k(self):
        return self._omega_tilde_k

    @omega_tilde_k.setter
    def omega_tilde_k(self, value):
        self._omega_tilde_k = atleast_2d_col(value)
