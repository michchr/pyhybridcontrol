from model_generators import DewhModelGenerator
from structdict import StructDict, StructDictAliased

from copy import deepcopy

import bisect

from sortedcontainers import SortedDict
import collections

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


class DewhRepository(DeviceRepository):
    def __init__(self, dewh_model_generator=None):
        device_type = 'dewh'
        super(DewhRepository, self).__init__(device_type, dewh_model_generator)

    def _device_creator(self, device_param_struct, dev_id):
        return DewhSys(self.model_generator, device_param_struct, dev_id)


class DewhSys(object):
    def __init__(self, model_generator, dewh_param_struct, dewh_id):
        self.dewh_param_struct = deepcopy(dewh_param_struct)
        self.dewh_id = dewh_id
        self.mld_mat_struct = StructDictAliased(A_h=[], B_h1=[], B_h2=[], B_h3=[], B_h4=[], b_h5=[], E_h1=[], E_h2=[],
                                                E_h3=[], E_h4=[], E_h5=[], d_h=[])
        self.var_dim_struct = model_generator.var_dim_struct
        self.V_offs = None
        self.W_offs = None
        self.d_offs = None

        self.initialize_sys(model_generator, dewh_param_struct)

    def initialize_sys(self, model_generator, dewh_param_struct):
        for mat_name, mat_eval in model_generator.mld_eval_struct.items():
            self.mld_mat_struct[mat_name] = mat_eval(dewh_param_struct)
            # self.mld_mat_struct[mat_name.replace("_eval", "")] = mat_eval(dewh_param_struct)
        self.set_offsets()

    def set_offsets(self):
        u_s = self._get_offset('nu')
        delta_s = self._get_offset('ndelta')
        z_s = self._get_offset('nz')
        omega_s = self._get_offset('nomega')
        d_s = self._get_offset('ncons')
        self.V_offs = V_h_s_ind(u_s, delta_s, z_s)
        self.W_offs = W_h_s_ind(omega_s)
        self.d_offs = d_h_s_ind(d_s)

    def _get_offset(self, arg):
        return (self.var_dim_struct[arg] if self.var_dim_struct[arg] else None)


if __name__ == '__main__':
    import timeit
    import pprint
    from parameters import dewh_p, grid_p


    def main():
        N_h = 1
        N_p = 3

        dewh_repo = DewhRepository(DewhModelGenerator)
        dewh_repo.default_param_struct = dewh_p

        for i in range(N_h):
            dewh_repo.add_device_by_default_data(i)

        sys = dewh_repo[0].mld_mat_struct


    def func():
        def closure():
            main()
            return 1

        return closure


    t1 = timeit.timeit(func(), number=1)
    print(t1)
