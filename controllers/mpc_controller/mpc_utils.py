from utils.func_utils import ParNotSet
from utils.matrix_utils import get_mat_ops


def process_base_args(self, f_kwargs=None, *,
                      N_p=ParNotSet, N_tilde=ParNotSet,
                      mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet, mld_info_k=ParNotSet, **kwargs):
    if N_p is None:
        f_kwargs['N_p'] = N_p = self.N_p

    if N_tilde is None:
        f_kwargs['N_tilde'] = N_p + 1 if N_p is not ParNotSet else self.N_p + 1

    if mld_numeric_tilde is None:
        f_kwargs['mld_numeric_tilde'] = mld_numeric_tilde = self.mld_numeric_tilde

    if mld_numeric_k is None:
        f_kwargs['mld_numeric_k'] = mld_numeric_k = self.mld_numeric_k

    if mld_info_k is None:
        mld_numeric_tilde = mld_numeric_tilde if mld_numeric_tilde is not ParNotSet else self.mld_numeric_tilde
        mld_numeric_k = mld_numeric_k if mld_numeric_k is not ParNotSet else self.mld_numeric_k
        f_kwargs['mld_info_k'] = mld_numeric_tilde[0].mldinfo if mld_numeric_tilde else mld_numeric_k.mld_info

    return f_kwargs


def process_mat_op_args(self, f_kwargs=None, *,
                        sparse=ParNotSet, mat_ops=ParNotSet, **kwargs):
    if sparse is None:
        f_kwargs['sparse'] = False

    if mat_ops is None:
        f_kwargs['mat_ops'] = get_mat_ops(sparse=f_kwargs['sparse'])

    return f_kwargs


def process_A_pow_tilde_arg(self, f_kwargs=None, *,
                            A_pow_tilde=ParNotSet, **kwargs):
    if A_pow_tilde is None:
        f_kwargs['A_pow_tilde'] = (
            self._gen_A_pow_tilde(_disable_process_args=True, N_tilde=f_kwargs['N_tilde'],
                                  mld_numeric_k=f_kwargs['mld_numeric_k'],
                                  mld_numeric_tilde=f_kwargs['mld_numeric_tilde'],
                                  sparse=f_kwargs['sparse'], mat_ops=f_kwargs['mat_ops']))

    return f_kwargs
