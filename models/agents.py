import bisect
from collections import OrderedDict
from reprlib import recursive_repr as _recursive_repr

# import pandas as pd
# pd.set_option('mode.chained_assignment', 'raise')

from controllers.mpc_controller.mpc_controller import MpcController
from structdict import StructDict, struct_repr
from models.mld_model import MldSystemModel


class Agent:
    _device_type_id_struct = StructDict()

    def __init__(self, device_type=None, device_id=None, sim_model: MldSystemModel = None):

        self._device_type = device_type or "not_specified"
        self._device_id = device_id
        self._sim_model = sim_model

        if self._device_type in self._device_type_id_struct:
            _id_set = self._device_type_id_struct[self._device_type].id_set
            _id_list = self._device_type_id_struct[self._device_type].id_list
            if self.device_id in _id_set:
                raise ValueError(
                    "Agent with type:'{}' and device_id:'{}' already exists".format(self._device_type, self.device_id))
            elif self.device_id is None:
                self._device_id = (_id_list[-1] + 1) if _id_list else 1

            _id_set.add(self._device_id)
            bisect.insort(_id_list, self._device_id)
        else:
            if self.device_id is None:
                self._device_id = 1
            self._device_type_id_struct[self._device_type] = StructDict(id_set=set(), id_list=[])
            self._device_type_id_struct[self._device_type].id_set.add(self._device_id)
            self._device_type_id_struct[self._device_type].id_list.append(self._device_id)

    # todo think about cleanup
    def __del__(self):
        # print("deleting")
        for col in self._device_type_id_struct[self._device_type].values():
            try:
                col.remove(self._device_id)
            except Exception:
                pass

    @property
    def device_type(self):
        return self._device_type

    @property
    def device_id(self):
        return self._device_id

    @property
    def sim_model(self):
        return self._sim_model

    @property
    def mld_numeric(self):
        return self._sim_model._mld_numeric

    @property
    def mld_numeric_tilde(self):
        return None

    @_recursive_repr()
    def __repr__(self):
        repr_dict = OrderedDict(device_type=self.device_type, device_id=self.device_id,
                                sim_model=self._sim_model)
        return struct_repr(repr_dict, type_name=self.__class__.__name__)


class MpcAgent(Agent):

    def __init__(self, device_type=None, device_id=None, sim_model=None, control_model=None, N_p=None, N_tilde=None):
        super().__init__(device_type=device_type, device_id=device_id, sim_model=sim_model)

        if all((sim_model, control_model)):
            self._sim_model = sim_model
            self._control_model = control_model
        else:
            self._sim_model = sim_model or control_model
            self._control_model = control_model or sim_model

        N_p = N_p if N_p is not None else 0
        N_tilde = N_tilde if N_tilde is not None else N_p + 1
        self._mpc_controller = MpcController(agent=self, N_p=N_p, N_tilde=N_tilde)

    @property
    def mpc_controller(self) -> MpcController:
        return self._mpc_controller

    @property
    def x_k(self):
        return self._mpc_controller.x_k

    @x_k.setter
    def x_k(self, x_k):
        self._mpc_controller.x_k = x_k

    @property
    def omega_tilde_k_hat(self):
        return self._mpc_controller.omega_tilde_k

    @omega_tilde_k_hat.setter
    def omega_tilde_k_hat(self, omega_tilde_k_hat):
        self._mpc_controller.omega_tilde_k = omega_tilde_k_hat

    def feedback(self, x_k=None, omega_tilde_k=None,
                 solver=None,
                 ignore_dcp=False, warm_start=True, verbose=False,
                 parallel=False, *, method=None, **kwargs
                 ):
        return self._mpc_controller.feedback(x_k=x_k, omega_tilde_k=omega_tilde_k,
                                             solver=solver,
                                             ignore_dcp=ignore_dcp, warm_start=warm_start, verbose=verbose,
                                             parallel=parallel, method=method, **kwargs
                                             )

    @property
    def sim_model(self):
        return self._sim_model

    @property
    def control_model(self):
        return self._control_model

    @property
    def mld_numeric_tilde(self):
        return None
