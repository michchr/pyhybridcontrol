from sortedcontainers import SortedDict
from copy import deepcopy
import collections

from utils.structdict import StructDict

V_h_s_ind = collections.namedtuple('V_ind', ['u_s', 'delta_s', 'z_s'])
W_h_s_ind = collections.namedtuple('W_ind', ['omega_s'])
d_h_s_ind = collections.namedtuple('d_ind', ['d_s'])

class DeviceRepository(SortedDict):
    def __init__(self, device_type, device_model_generator=None):
        super(DeviceRepository, self).__init__()
        self.repository = self

        if device_model_generator:
            self.model_generator = device_model_generator()
            self._required_param_list = self.model_generator.required_param_list
            self._default_param_struct = StructDict.fromkeys(self._required_param_list)
        else:
            self.model_generator = None
            self._required_param_list = None
            self._default_param_struct = None

        self.device_type = device_type
        self._N_dev = 0

    @property
    def N_dev(self):
        return self._N_dev

    @property
    def default_param_struct(self):
        return self._default_param_struct

    @default_param_struct.setter
    def default_param_struct(self, device_param_struct):
        if isinstance(device_param_struct, StructDict):
            if set(set(self._required_param_list)).issubset(set(device_param_struct.keys())):
                self._default_param_struct = device_param_struct
            else:
                raise ValueError("All required parameters not included, need {}".format(self._required_param_list))
        else:
            raise ValueError("Invalid argument, requires type==Struct_dict")

    #### Methods for adding devices to repository

    def add_device_by_default_data(self, dev_id=None):
        self._add_device(self.default_param_struct, dev_id=dev_id)

    def _add_device(self, device_param_struct, dev_id=None):
        if dev_id == None:
            dev_id = self.repository.keys()[-1] + 1 if self.repository.keys() else 1
        elif dev_id in self.repository.keys():
            raise ValueError("Id {} already exists in dewh_repository".format(dev_id))
        elif not self.default_param_struct:
            raise ValueError(
                "Default parameter list not set, need Struct_dict with {}".format(self._required_param_list))

        self.repository[dev_id] = self._device_creator(device_param_struct, dev_id)
        self._N_dev += 1

    def _device_creator(self, device_param_struct, dev_id):
        return dev_id
