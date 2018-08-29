import scipy.sparse as scs
import numpy as np
from datetime import datetime as DateTime
import pprint

from models.device_repository import DeviceRepository, DewhRepository
from utils.structdict import StructDict, StructDictAliased
from models.model_generators import DewhModelGenerator, GridModelGenerator


class MicroGridModel():

    def __init__(self):
        self.device_repositories = []
        self._device_mld_mat_struct = None

        self._grid_param_struct = None
        self._grid_mld_mat_struct = None
        self._grid_var_dims_struct = None

        self._decision_var_types = None
        self._date_time_0 = None

    @property
    def date_time_0(self):
        return self._date_time_0

    @date_time_0.setter
    def date_time_0(self, date_time_0):
        if isinstance(date_time_0, DateTime):
            self._date_time_0 = date_time_0
        else:
            raise TypeError("date_time_0 not of type {type}".format(type=DateTime))

    @property
    def device_mld_mat_struct(self):
        return self._device_mld_mat_struct

    @property
    def grid_mld_mat_struct(self):
        return self._grid_mld_mat_struct

    @property
    def decision_var_types(self):
        return self._decision_var_types
        
    @property
    def grid_param_struct(self):
        return self._grid_param_struct

    @grid_param_struct.setter
    def grid_param_struct(self, grid_param_struct):
        self._grid_param_struct = grid_param_struct


    def add_device_repository(self, device_repository: DeviceRepository):
        self.device_repositories.append(device_repository)

    def gen_concat_device_system_mld(self, sparse=True):
        mld_mat_list = StructDictAliased(A_s=[], B_s1=[], B_s2=[], B_s3=[], B_s4=[], b_s5=[], E_s1=[], E_s2=[],
                                         E_s3=[], E_s4=[], E_s5=[], d_s=[])

        # First extract matrices form repository as list
        for device_repository in self.device_repositories:
            for device_id, device in device_repository.items():
                for mld_mat_id, mld_mat in device.mld_mat_struct.items():
                    mld_mat_list[mld_mat_id].append(mld_mat)

        # Then generate sparse matrices
        mld_mat_struct = StructDict()
        for key, value in mld_mat_list.items():
            if key[0].isupper():
                mld_mat_struct[key] = scs.block_diag(value)
            else:
                mld_mat_struct[key] = np.vstack(value)

        self._device_mld_mat_struct = mld_mat_struct

        return mld_mat_struct

    def gen_power_balance_constraint_mld(self, grid_param_struct=None):
        if grid_param_struct == None:
            grid_param_struct = self.grid_param_struct

        grid_mld_mat_struct = StructDictAliased(A_p=[], B_p1=[], B_p2=[], B_p3=[], B_p4=[], b_p5=[], E_p1=[],
                                                     E_p2=[], E_p3=[], E_p4=[], E_p5=[], d_p=[])
        grid_model_generator = GridModelGenerator()

        self._grid_var_dims_struct = grid_model_generator.var_dim_struct

        for mat_name, mat_eval in grid_model_generator.mld_eval_struct.items():
            grid_mld_mat_struct[mat_name] = mat_eval(grid_param_struct)
            # self.mld_mat_struct[mat_name.replace("_eval", "")] = mat_eval(dewh_param_struct)

        self._grid_mld_mat_struct = grid_mld_mat_struct

        return grid_mld_mat_struct

    def get_grid_summation_vector(self):
        sum_load_list = []
        for device_repository in self.device_repositories:
            if device_repository.device_type == 'dewh':
                for dev_id, device in device_repository.items():
                    sum_load_list.append(device.dewh_param_struct.P_h_Nom)
            else:
                sum_load_list.append(1)

        zeros_delta_z = np.zeros((0)) # NEEDS TO BE UPDATED TO INCLUDE BATTERIES ETC!!!!!
        sum_load_vec = np.atleast_2d(np.hstack([sum_load_list, zeros_delta_z])).T

        return sum_load_vec

    def get_decision_var_type(self):
        decision_var_types_list = []
        # First extract matrices form repository as list
        for device_repository in self.device_repositories:
            for device_id, device in device_repository.items():
                decision_var_types_list.append(device.var_dim_struct.decision_var_types)

        decision_var_types_list.append(self._grid_var_dims_struct.decision_var_types)

        decision_var_types = np.vstack(decision_var_types_list)

        self._decision_var_types = decision_var_types

        return decision_var_types




if __name__ == '__main__':
    import timeit
    from models.parameters import dewh_p, grid_p

    def main():
        N_h = 1


        dewh_repo = DewhRepository(DewhModelGenerator)
        dewh_repo.default_param_struct = dewh_p

        for i in range(N_h):
            # dewh_repo.default_param_struct.P_h_Nom +=1
            dewh_repo.add_device_by_default_data(i)

        mg_model = MicroGridModel()
        mg_model.grid_param_struct = grid_p

        mg_model.add_device_repository(dewh_repo)
        mg_model.gen_concat_device_system_mld()

        mg_model.gen_power_balance_constraint_mld(grid_p)
        #
        pprint.pprint(mg_model.grid_mld_mat_struct)
        # pprint.pprint(mg_model.device_mld_mat_struct)

        pprint.pprint(mg_model.get_decision_var_type())

        print(mg_model.get_grid_summation_vector())

    def func():
        def closure():
            main()
            return 1

        return closure


    t1 = timeit.timeit(func(), number=1)
    print(t1)
