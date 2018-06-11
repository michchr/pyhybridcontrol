import model_generators as mg
import struct_dict as sd

import bisect
import sortedcontainers as scon
import collections

V_h_s_ind = collections.namedtuple('V_ind', ['u_s', 'delta_s', 'z_s'])
W_h_s_ind = collections.namedtuple('W_ind', ['omega_s'])
d_h_s_ind = collections.namedtuple('d_ind', ['d_s'])


class Dewh_Repository():
    def __init__(self, dewh_model_generator=mg.dewh_model_generator):
        self.repository = scon.SortedDict()  # Empty dictionary for storing individual water heaters
        self.model_generator = dewh_model_generator()

        self._N_h = 0  # Number of water heaters
        self._required_param_list = self.model_generator.required_param_list
        self._default_dewh_param_struct = sd.Struct_Dict.fromkeys(self._required_param_list)

    @property
    def N_h(self):
        return self._N_h

    @property
    def default_dewh_param_struct(self):
        return self._default_dewh_param_struct

    @default_dewh_param_struct.setter
    def default_dewh_param_struct(self, dewh_param_struct):
        if isinstance(dewh_param_struct, sd.Struct_Dict):
            if set(set(self._required_param_list)).issubset(set(dewh_param_struct.keys())):
                self._default_dewh_param_struct = dewh_param_struct
            else:
                raise ValueError("All required parameters not included, need %s" % self._required_param_list)
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

        self.repository[dewh_id] = Dewh_Sys(self.model_generator, dewh_param_struct, dewh_id)
        self._N_h += 1




class Dewh_Sys():
    def __init__(self, model_generator, dewh_param_struct, dewh_id):
        self.dewh_param_struct = dewh_param_struct
        self.dewh_id = dewh_id
        self.sys_mats = {}
        self.con_mats = {}
        self.var_dim_struct = model_generator.var_dim_struct
        self.V_offs = None
        self.W_offs = None
        self.d_offs = None

        self.initialize_sys(model_generator, dewh_param_struct)

    def initialize_sys(self, model_generator, dewh_param_struct):
        for mat_name, mat_eval in model_generator.sys_matrix_eval_dict.items():
            self.sys_mats[mat_name.replace("_eval", "")] = mat_eval(dewh_param_struct)

        for mat_name, mat_eval in model_generator.con_matrix_eval_dict.items():
            self.con_mats[mat_name.replace("_eval", "")] = mat_eval(dewh_param_struct)

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
    import random

    dewh_repo = Dewh_Repository(mg.dewh_model_generator)
    # print(dewh_repo.required_param_list)
    # print(dewh_repo.model_generator.sys_matrix_eval_dict.get('A_h_eval').__doc__)
    # print(dewh_repo.default_dewh_params)

    control_ts = 15 * 60

    dewh_p = sd.Struct_Dict()
    dewh_p.C_w = 4.1816 * 10 ** 3
    dewh_p.A_h = 1
    dewh_p.U_h = 2.7
    dewh_p.m_h = 150
    dewh_p.T_w = 15
    dewh_p.T_inf = 20
    dewh_p.P_h_Nom = 3000
    dewh_p.T_h_min = 40
    dewh_p.T_h_max = 65
    dewh_p.ts = control_ts

    # print(set(dewh_p.keys()))

    dewh_repo.default_dewh_param_struct = dewh_p
    # print(dewh_repo._default_dewh_param_struct)
    # for i in range(1000):
    #     dewh_repo.add_dewh_by_default_data(i)

    # print(dewh_repo.repository.get(99).sys_mats)
    import scipy.linalg as scl
    import scipy.sparse as scs
    import random

    a_rnd = random.sample(range(10), 10)

    b = [i for i in range(1, 10000)]


    def func():
        def closure():
            for i in a_rnd:
                dewh_repo.add_dewh_by_default_data(i)
            a, b = dewh_repo.get_overall_dewh_sys_matrices()
            print(a)

            # print(dewh_repo.repository.get(1).V_offs)

            return 1

        return closure


    t1 = timeit.timeit(func(), number=1)
    print(t1)
