from model_generators import DewhModelGenerator
from structdict import StructDict, StructDictAliased

import bisect

from sortedcontainers import SortedDict
import collections

V_h_s_ind = collections.namedtuple('V_ind', ['u_s', 'delta_s', 'z_s'])
W_h_s_ind = collections.namedtuple('W_ind', ['omega_s'])
d_h_s_ind = collections.namedtuple('d_ind', ['d_s'])


class DewhRepository(SortedDict):
    def __init__(self, dewh_model_generator=DewhModelGenerator):
        super(DewhRepository, self).__init__()
        self.repository = self  # Empty dictionary for storing individual water heaters
        self.model_generator = dewh_model_generator()
        self.device_type = 'dewh'
        self._N_h = 0  # Number of water heaters
        self._required_param_list = self.model_generator.required_param_list
        self._default_dewh_param_struct = StructDict.fromkeys(self._required_param_list)
        self._default_dewh_param_struct_set = False

    @property
    def N_h(self):
        return self._N_h

    @property
    def default_dewh_param_struct(self):
        return self._default_dewh_param_struct

    @default_dewh_param_struct.setter
    def default_dewh_param_struct(self, dewh_param_struct):
        if isinstance(dewh_param_struct, StructDict):
            if set(set(self._required_param_list)).issubset(set(dewh_param_struct.keys())):
                self._default_dewh_param_struct = dewh_param_struct
                self._default_dewh_param_struct_set = True
            else:
                raise ValueError("All required parameters not included, need {}".format(self._required_param_list))
        else:
            raise ValueError("Invalid argument, requires type==Struct_dict")

    def add_dewh_by_default_data(self, dewh_id=None):
        self._add_dewh(self._default_dewh_param_struct, dewh_id)

    def add_dewh_by_custom_data(self, dewh_param_struct, dewh_id=None):
        self._add_dewh(dewh_param_struct, dewh_id)

    def update_dewh_param(self, dewh_param_struct, dewh_id):
        pass

    def _add_dewh(self, dewh_param_struct, dewh_id=None):
        if dewh_id == None:
            dewh_id = self.repository.keys()[-1] + 1 if self.repository.keys() else 1
        elif dewh_id in self.repository.keys():
            raise ValueError("Id {} already exists in dewh_repository".format(dewh_id))
        elif not self._default_dewh_param_struct_set:
            raise ValueError(
                "Default parameter list not set, need Struct_dict with {}".format(self._required_param_list))

        self.repository[dewh_id] = DewhSys(self.model_generator, dewh_param_struct, dewh_id)
        self._N_h += 1

    def get_dewh_dyn_con(self, dewh_id):
        dewh = self.get(dewh_id)
        return dewh.mld_mat_stuct


class DewhSys(object):
    def __init__(self, model_generator, dewh_param_struct, dewh_id):
        self.dewh_param_struct = dewh_param_struct
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
    from parameters import dewh_p, grid_p
    import timeit
    import pprint
    import random

    dewh_repo = DewhRepository(DewhModelGenerator)



    dewh_repo.default_dewh_param_struct = dewh_p

    import random

    a_rnd = random.sample(range(1000), 1000)

    b = [i for i in range(1, 10000)]


    def func():
        def closure():
            for i in a_rnd:
                dewh_repo.add_dewh_by_default_data(i)
            # print(dewh_repo.get(2).mld_mat_struct)
            return 1

        return closure


    t1 = timeit.timeit(func(), number=1)
    print(t1)
