from models.micro_grid_models import DewhModel, GridModel
from models.agents import MpcAgent

from utils.func_utils import ParNotSet

import cvxpy as cvx


class DewhAgentMpc(MpcAgent):

    def __init__(self, device_type='dewh', device_id=None,
                 sim_model=ParNotSet, control_model=ParNotSet,
                 N_p=None, N_tilde=None):
        sim_model = sim_model if sim_model is not ParNotSet else DewhModel(const_heat=False)
        control_model = control_model if control_model is not ParNotSet else DewhModel(const_heat=True)

        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=sim_model, control_model=control_model,
                         N_p=N_p, N_tilde=N_tilde)

    @property
    def power_N_tilde(self):
        return self._mpc_controller.variables.u.var_N_tilde * self.control_model.param_struct.P_h_Nom

    @property
    def cost(self):
        return self._mpc_controller.cost

    @property
    def constraints(self):
        return self._mpc_controller.constraints

    def set_device_penalties(self, objective_weights_struct=None, **kwargs):
        self._mpc_controller.set_std_obj_weights(objective_atoms_struct=objective_weights_struct, **kwargs)

    def build_device(self, with_std_cost=True, with_std_constraints=True, sense=None, disable_soft_constraints=False):
        self._mpc_controller.build(with_std_cost=with_std_cost, with_std_constraints=with_std_constraints, sense=sense,
                                   disable_soft_constraints=disable_soft_constraints)


class GridAgentMpc(MpcAgent):

    def __init__(self, device_type='grid', device_id=None, N_p=None, N_tilde=None):

        grid_model = GridModel(num_devices=0)
        super().__init__(device_type=device_type, device_id=device_id,
                         sim_model=grid_model, control_model=grid_model,
                         N_p=N_p, N_tilde=N_tilde)

        self.devices = []
        self.device_powers = []
        self.device_constraints = []
        self.device_costs = []

    def add_device(self, device):
        self.devices.append(device)

    def build_grid(self, N_p=ParNotSet, N_tilde=ParNotSet):
        self.update_horizons(N_p=N_p, N_tilde=N_tilde)
        for device in self.devices:
            if device.N_tilde != self.N_tilde:
                raise ValueError(
                    f"All devices in self.devices must have 'N_tilde' equal to 'self.N_tilde':{self.N_tilde}")
        grid_model = GridModel(num_devices=len(self.devices))
        self.update_models(sim_model=grid_model, control_model=grid_model)

    def get_device_data(self):
        device_powers = []
        device_costs = []
        device_constraints = []

        for device in self.devices:
            device_powers.append(device.power_N_tilde)
            device_costs.append(device.cost)
            device_constraints.extend(device.constraints)

        self.device_powers = device_powers
        self.device_costs = device_costs
        self.device_constraints = device_constraints

    def set_grid_device_powers(self):
        omega_mat_tilde_k = cvx.hstack(self.device_powers).T
        self._mpc_controller.omega_tilde_k = cvx.reshape(omega_mat_tilde_k, (omega_mat_tilde_k.size, 1))

    def set_grid_device_costs(self):
        self._mpc_controller.set_costs(other_costs=self.device_costs)

    def set_grid_device_constraints(self):
        self._mpc_controller.set_constraints(other_constraints=self.device_constraints)


if __name__ == '__main__':
    pass
