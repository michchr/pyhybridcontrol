import bisect
from collections import OrderedDict
from reprlib import recursive_repr as _recursive_repr

# import pandas as pd
# pd.set_option('mode.chained_assignment', 'raise')

from controllers.mpc_controller.mpc_controller import MpcController
from structdict import StructDict, struct_repr
from models.mld_model import MldSystemModel


class Agent:
    _agent_type_id_struct = StructDict()

    def __init__(self, agent_type=None, agent_id=None, agent_model: MldSystemModel = None):

        self._agent_type = agent_type or "not_specified"
        self._agent_id = agent_id
        self._agent_model = agent_model

        if self._agent_type in self._agent_type_id_struct:
            _id_set = self._agent_type_id_struct[self._agent_type].id_set
            _id_list = self._agent_type_id_struct[self._agent_type].id_list
            if self.agent_id in _id_set:
                raise ValueError(
                    "Agent with type:'{}' and agent_id:'{}' already exists".format(self._agent_type, self.agent_id))
            elif self.agent_id is None:
                self._agent_id = (_id_list[-1] + 1) if _id_list else 1

            _id_set.add(self._agent_id)
            bisect.insort(_id_list, self._agent_id)
        else:
            if self.agent_id is None:
                self._agent_id = 1
            self._agent_type_id_struct[self._agent_type] = StructDict(id_set=set(), id_list=[])
            self._agent_type_id_struct[self._agent_type].id_set.add(self._agent_id)
            self._agent_type_id_struct[self._agent_type].id_list.append(self._agent_id)

    # todo think about cleanup
    def __del__(self):
        # print("deleting")
        for col in self._agent_type_id_struct[self._agent_type].values():
            try:
                col.remove(self._agent_id)
            except Exception:
                pass

    @property
    def agent_type(self):
        return self._agent_type

    @property
    def agent_id(self):
        return self._agent_id

    @property
    def agent_model(self):
        return self._agent_model

    @property
    def mld_numeric(self):
        return self._agent_model._mld_numeric

    @property
    def mld_numeric_tilde(self):
        return None

    @_recursive_repr()
    def __repr__(self):
        repr_dict = OrderedDict(agent_type=self.agent_type, agent_id=self.agent_id,
                                agent_model=self._agent_model)
        return struct_repr(repr_dict, type_name=self.__class__.__name__)


class MpcAgent(Agent):

    def __init__(self, agent_type=None, agent_id=None, agent_model=None, N_p=None):
        super().__init__(agent_type=agent_type, agent_id=agent_id, agent_model=agent_model)
        self._N_p = N_p if N_p is not None else 0
        self._N_tilde = self._N_p + 1
        self._mpc_controller = MpcController(self)

    @property
    def mpc_controller(self)->MpcController:
        return self._mpc_controller

    @property
    def N_p(self):
        return self._N_p

    @property
    def N_tilde(self):
        return self._N_tilde

    @property
    def mld_numeric_tilde(self):
        return None
