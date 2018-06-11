
# def get_overall_dewh_sys_matrices(self):
#     sys_mat_dict = {'A_h_s': [], 'B_h1_s': [], 'B_h2_s': [], 'B_h3_s': [], 'B_h4_s': [], 'b_h5_s': []}
#     con_mat_dict = {'E_h1_s': [], 'E_h2_s': [], 'E_h3_s': [], 'E_h4_s': [], 'E_h5_s': [], 'd_h_s': []}
#
#     for dewh_id, dewh_sys in self.repository.items():
#         for sys_mat_id, sys_mat in dewh_sys.sys_mats.items():
#             sys_mat_dict[sys_mat_id + "_s"].append(sys_mat)
#         for con_mat_id, con_mat in dewh_sys.con_mats.items():
#             con_mat_dict[con_mat_id + "_s"].append(con_mat)
#
#     return sys_mat_dict, con_mat_dict