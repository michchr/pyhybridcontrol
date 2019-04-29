import bisect
from collections import OrderedDict
from reprlib import recursive_repr as _recursive_repr

# import pandas as pd
# pd.set_option('mode.chained_assignment', 'raise')

from controllers.mpc_controller import MpcController
from controllers.controller_base import ControllerBase
from structdict import StructDict, struct_repr, named_struct_dict
from models.mld_model import MldSystemModel, MldModel, MldInfo

from utils.func_utils import ParNotSet
from utils.helper_funcs import is_all_None

from typing import MutableMapping, AnyStr


class Agent:
    _device_type_id_struct = StructDict()

    def __init__(self, device_type=None, device_id=None,
                 sim_model: MldSystemModel = ParNotSet,
                 control_model: MldSystemModel = ParNotSet):

        self._device_type = None
        self._device_id = None
        self._sim_model = None
        self._control_model = None

        self.update_device_data(device_type=device_type, device_id=device_id)
        self.update_models(sim_model=sim_model, control_model=control_model)

    def update_device_data(self, device_type=None, device_id=None):
        self._device_type = device_type if device_type is not None else self._device_type or 'not_specified'
        self._device_id = device_id if device_id is not None else self._device_id

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

    def update_models(self, sim_model: MldSystemModel = ParNotSet,
                      control_model: MldSystemModel = ParNotSet):

        if is_all_None(self._sim_model, self._control_model, sim_model, control_model):
            self._sim_model = MldSystemModel()
            self._control_model = None
        else:
            self._sim_model = sim_model if sim_model is not ParNotSet else self._sim_model or MldSystemModel()
            self._control_model = control_model if control_model is not ParNotSet else self._control_model or None

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
    def sim_model(self) -> MldSystemModel:
        return self._sim_model

    @property
    def control_model(self) -> MldSystemModel:
        return self._control_model if self._control_model is not None else self._sim_model

    @property
    def mld_numeric(self) -> MldModel:
        return self._sim_model._mld_numeric

    @property
    def mld_info(self) -> MldInfo:
        return self.mld_numeric.mld_info

    @property
    def mld_numeric_tilde(self):
        return None

    @_recursive_repr()
    def __repr__(self):
        repr_dict = OrderedDict(device_type=self.device_type,
                                device_id=self.device_id,
                                sim_model=self.sim_model,
                                control_model=self.control_model)
        return struct_repr(repr_dict, type_name=self.__class__.__name__)


class ControlledAgent(Agent):
    ControllersStruct = named_struct_dict('ControllersStruct')

    def __init__(self, device_type=None, device_id=None, sim_model=None, control_model=None, N_p=None, N_tilde=None):
        self._controllers: MutableMapping[AnyStr, ControllerBase] = self.ControllersStruct()
        super().__init__(device_type=device_type, device_id=device_id, sim_model=sim_model, control_model=control_model)


    def update_models(self, sim_model: MldSystemModel = ParNotSet,
                      control_model: MldSystemModel = ParNotSet):
        super(ControlledAgent, self).update_models(sim_model=sim_model, control_model=control_model)
        for controller in self.controllers.values():
            controller.reset_components()

    @property
    def controllers(self):
        return self._controllers

    def add_controller(self, name, controller, x_k=None, omega_tilde_k=None, N_p=None, N_tilde=None):
        self._controllers[name] = controller(agent=self, x_k=x_k, omega_tilde_k=omega_tilde_k, N_p=N_p, N_tilde=N_tilde)


class MpcAgent(Agent):

    def __init__(self, device_type=None, device_id=None, sim_model=None, control_model=None, N_p=None, N_tilde=None):
        super().__init__(device_type=device_type, device_id=device_id, sim_model=sim_model, control_model=control_model)

        self._mpc_controller = MpcController(agent=self, N_p=N_p, N_tilde=N_tilde)

    def update_models(self, sim_model: MldSystemModel = ParNotSet,
                      control_model: MldSystemModel = ParNotSet):
        super(MpcAgent, self).update_models(sim_model=sim_model, control_model=control_model)
        mpc_controller: MpcController = getattr(self, '_mpc_controller', None)
        if mpc_controller:
            mpc_controller.reset_components()

    def update_horizons(self, N_p=ParNotSet, N_tilde=ParNotSet):
        N_p = N_p if N_p is not ParNotSet else self.N_p or 0
        N_tilde = N_tilde if N_tilde is not ParNotSet else N_p + 1
        self._mpc_controller.update_horizons(N_p=N_p, N_tilde=N_tilde)

    @property
    def mpc_controller(self) -> MpcController:
        return self._mpc_controller

    @property
    def N_p(self):
        return self._mpc_controller.N_p if self._mpc_controller else None

    @property
    def N_tilde(self):
        return self._mpc_controller.N_tilde if self._mpc_controller else None

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
