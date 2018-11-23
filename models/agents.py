import bisect
import inspect
import itertools
from collections import OrderedDict
from copy import copy as _copy
from reprlib import recursive_repr as _recursive_repr
import types

import functools

from enum import IntEnum

import numpy as np
import pandas as pd
import scipy.linalg as scl
import scipy.sparse as scs
import wrapt
from numpy.lib.stride_tricks import as_strided as _as_strided

from models.mld_model import MldModel, MldModelTypes
from utils.structdict import StructDict, struct_repr

pd.set_option('mode.chained_assignment', 'raise')


class AgentModelGenerator:

    def __init__(self, *args, **kwargs):
        self.mld_symbolic = self.get_mld_symbolic(*args, **kwargs)
        self.mld_eval = self.get_mld_eval()

    def get_mld_symbolic(self, *args, **kwargs):
        raise NotImplementedError("Need to implement symbolic mld")

    def get_mld_eval(self, symbolic_mld=None):
        symbolic_mld = symbolic_mld or self.mld_symbolic
        return symbolic_mld.to_eval()

    def get_mld_numeric(self, param_struct=None, mld=None, copy=True):
        mld = mld or self.mld_eval or self.mld_symbolic
        return mld.to_numeric(param_struct=param_struct, copy=copy)

    def get_required_params(self):
        return self.mld_symbolic.mld_info['required_params']


class Agent:
    _agent_type_id_struct = StructDict()

    def __init__(self, agent_type=None, agent_id=None, agent_model_generator: AgentModelGenerator = None,
                 param_struct=None, mld_numeric: MldModel = None):

        self._mld_numeric = mld_numeric
        self._agent_model_generator = agent_model_generator
        self._param_struct = {}
        self._param_struct = self._validate_param_struct(param_struct=param_struct, missing_param_check=True,
                                                         invalid_param_check=False)
        if self._agent_model_generator is not None:
            self._symbolic_mld = agent_model_generator.mld_symbolic
            self._eval_mld = agent_model_generator.mld_eval
            self._mld_numeric = self._mld_numeric or agent_model_generator.get_mld_numeric(
                param_struct=self._param_struct)

        self._agent_type = agent_type or "not_specified"
        self._agent_id = agent_id

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
        print("deleting")
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
    def agent_model_generator(self):
        return self._agent_model_generator

    @property
    def mld_numeric(self):
        return self._mld_numeric

    @property
    def param_struct(self):
        return _copy(self._param_struct)

    @param_struct.setter
    def param_struct(self, param_struct):
        self.update_param_struct(param_struct=param_struct)

    def update_param_struct(self, param_struct=None, param_struct_subset=None, missing_param_check=True,
                            invalid_param_check=False, **kwargs):
        self._param_struct = self._validate_param_struct(param_struct=param_struct,
                                                         param_struct_subset=param_struct_subset,
                                                         missing_param_check=missing_param_check,
                                                         invalid_param_check=invalid_param_check, **kwargs)

    def _validate_param_struct(self, param_struct=None, param_struct_subset=None, missing_param_check=False,
                               invalid_param_check=False, **kwargs):
        param_struct_subset = param_struct_subset if param_struct_subset is not None else {}
        param_struct = param_struct if param_struct is not None else self._param_struct
        try:
            param_struct_subset.update(kwargs)
        except AttributeError:
            raise TypeError("Invalid type for 'param_struct_subset', must be dictionary like or None.")

        if not param_struct_subset and param_struct is self._param_struct:
            return self._param_struct
        elif not param_struct or param_struct is self._param_struct:
            given_params = param_struct_subset
            param_struct = _copy(self._param_struct)
            param_struct.update(param_struct_subset)
        else:
            try:
                param_struct.update(param_struct_subset)
            except AttributeError:
                raise TypeError("Invalid type for 'param_struct', must be dictionary like or None.")
            given_params = param_struct

        if missing_param_check:
            try:
                required_params = set(self._agent_model_generator.get_required_params())
            except AttributeError:
                required_params = set()

            if required_params:
                missing_keys = required_params.difference(param_struct.keys())
                if missing_keys:
                    raise ValueError(
                        "The following keys are missing from the given param_struct: '{}'".format(missing_keys)
                    )

        if invalid_param_check:
            invalid_params = set(given_params.keys()).difference(self._param_struct.keys())
            if invalid_params:
                raise ValueError(
                    "Invalid keys:'{}' in kwargs/param_struct - keys must all exist in self.param_struct. Hint: "
                    "either disable 'invalid_param_check' or update self.param_struct.".format(invalid_params)
                )

        valid_param_struct = param_struct if isinstance(param_struct, StructDict) else StructDict(param_struct)
        return valid_param_struct

    def get_mld_numeric(self, param_struct=None, param_struct_subset=None, missing_param_check=False,
                        invalid_param_check=True, copy=True, **kwargs):

        compute_param_struct = self._validate_param_struct(param_struct=param_struct,
                                                           param_struct_subset=param_struct_subset,
                                                           missing_param_check=missing_param_check,
                                                           invalid_param_check=invalid_param_check,
                                                           **kwargs)

        if compute_param_struct:
            try:
                return self._agent_model_generator.get_mld_numeric(compute_param_struct, copy=copy)
            except AttributeError:
                raise TypeError("Agent does not contain valid agent_model_generator.")
        else:
            return self._mld_numeric

    def update_mld_numeric(self, param_struct=None, mld_numeric=None, copy=True, **kwargs):
        if isinstance(mld_numeric, MldModel) and mld_numeric.mld_type == MldModelTypes.numeric:
            self._mld_numeric = mld_numeric
            return self.mld_numeric
        else:
            self.update_param_struct(param_struct, **kwargs)

        try:
            return self._agent_model_generator.get_mld_numeric(param_struct=self._param_struct, copy=copy)
        except AttributeError:
            raise TypeError("Agent does not contain valid agent_model_generator.")

    @_recursive_repr()
    def __repr__(self):
        repr_dict = OrderedDict(agent_type=self.agent_type, agent_id=self.agent_id,
                                agent_model_generator=self._agent_model_generator,
                                mld_numeric=self._mld_numeric)
        return struct_repr(repr_dict, type_name=self.__class__.__name__)


def _get_cached_signature(func):
    if hasattr(func, '_f_signature'):
        f_signature = func._f_signature  # use cached _f_signature
    else:
        f_signature = inspect.signature(func)
        if isinstance(func, types.MethodType):
            func.__func__._f_signature = f_signature  # cache signature as function attribute
        else:
            func._f_signature = f_signature

    return f_signature


def _cache_hashable_args(maxsize=128, typed=False):
    def lru_wrapper(func):
        lru_wrapped = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        @wrapt.decorator(adapter=func)
        def wrapper(func, instance, args, kwargs):
            return func(*args, **kwargs)

        return wrapper(lru_wrapped)

    return lru_wrapper


@wrapt.decorator
def _process_args_mpc_decor(func, self, args_in, kwargs_in):
    f_signature = _get_cached_signature(func)
    f_kwargs_default = {param_name: param.default for param_name, param in f_signature.parameters.items() if
                        param.default is not inspect.Parameter.empty}

    if args_in:
        kw_update = {param_name: value for value, (param_name, param) in zip(args_in, f_signature.parameters.items()) if
                     param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD}
        args = args_in[:-len(kw_update)]
        kwargs_in.update(kw_update)
    else:
        args = ()

    kwargs = dict(f_kwargs_default, **kwargs_in)
    kwargs = self._process_mpc_func_args(f_kwargs=kwargs, f_signature=f_signature, *args, **kwargs)
    return func(*args, **kwargs)


class _ParamRequired(IntEnum):
    TRUE = True
    FALSE = False


_ParNotReq = _ParamRequired.FALSE
_ParReq = _ParamRequired.TRUE


class MpcAgent(Agent):

    def __init__(self, *args, N_p=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_p = N_p if N_p is not None else 0
        self.time_k0 = 0
        self.state_input_evolution_struct = StructDict(time_k0=None)
        self.con_evolution_struct = StructDict(time_k0=None)

    def _process_mpc_func_args(self, f_kwargs=None, f_signature=None, *args,
                               N_p=_ParNotReq, param_struct=_ParNotReq, param_struct_subset=_ParNotReq,
                               schedule_params_tilde=_ParNotReq, param_struct_tilde=_ParNotReq,
                               include_term_cons=_ParNotReq, mld_numeric=_ParNotReq, mld_numeric_tilde=_ParNotReq,
                               A_pow_tilde=_ParNotReq, sparse=_ParNotReq, mat_ops=_ParNotReq, copy=_ParNotReq,
                               **kwargs):

        if N_p is None:
            f_kwargs['N_p'] = N_p = self.N_p
        else:
            N_p = f_kwargs.get('N_p')

        if include_term_cons is None:
            f_kwargs['include_term_cons'] = include_term_cons = True
        else:
            include_term_cons = f_kwargs.get('include_term_cons')

        if sparse is None:
            f_kwargs['sparse'] = sparse = False
        else:
            sparse= f_kwargs.get('sparse')

        if copy is None:
            f_kwargs['copy'] = copy = True
        else:
            copy = f_kwargs.get('copy')

        if mat_ops is None:
            f_kwargs['mat_ops'] = mat_ops = self._get_mat_ops(sparse)
        else:
            mat_ops = f_kwargs.get('mat_ops')

        if param_struct is None:
            f_kwargs['param_struct'] = param_struct = (
                self._validate_param_struct(param_struct=self._param_struct, param_struct_subset=param_struct_subset,
                                            missing_param_check=False, invalid_param_check=False))
        else:
            param_struct = f_kwargs.get('param_struct')

        if (schedule_params_tilde or kwargs) and param_struct_tilde is None:
            f_kwargs['param_struct_tilde'] = param_struct_tilde = (
                self._gen_param_struct_tilde(N_p=N_p, param_struct=param_struct,
                                             param_struct_subset=param_struct_subset,
                                             schedule_params_tilde=schedule_params_tilde,
                                             include_term_cons=include_term_cons, **kwargs))
        else:
            param_struct_tilde = f_kwargs.get('param_struct_tilde')

        if param_struct_tilde and mld_numeric_tilde is None:
            f_kwargs['mld_numeric_tilde'] = mld_numeric_tilde = (
                self._gen_mld_numeric_tilde(N_p=N_p, param_struct=param_struct, param_struct_subset=param_struct_subset,
                                            schedule_params_tilde=schedule_params_tilde,
                                            param_struct_tilde=param_struct_tilde, include_term_cons=include_term_cons,
                                            sparse=sparse, copy=copy, **kwargs))
        else:
            mld_numeric_tilde = f_kwargs.get('mld_numeric_tilde')

        if mld_numeric is None:
            if param_struct is self._param_struct:
                mld_numeric = self.mld_numeric
            else:
                mld_numeric = self.get_mld_numeric(param_struct=param_struct, missing_param_check=False,
                                                   invalid_param_check=False, copy=copy)
            f_kwargs['mld_numeric'] = mld_numeric
        else:
            mld_numeric = f_kwargs.get('mld_numeric')

        if A_pow_tilde is None:
            f_kwargs['A_pow_tilde'] = (
                self._gen_A_pow_tilde(N_p=N_p, param_struct=param_struct, param_struct_subset=param_struct_subset,
                                      schedule_params_tilde=schedule_params_tilde,
                                      param_struct_tilde=param_struct_tilde, mld_numeric=mld_numeric,
                                      mld_numeric_tilde=mld_numeric_tilde, sparse=sparse, mat_ops=mat_ops, copy=copy,
                                      **kwargs))

        arg_valid_test = False
        if arg_valid_test:
            if f_kwargs.get('N_p') and f_kwargs['N_p'] < 0:
                raise ValueError("N_p must have a value of N_p >= 0.")
            if f_kwargs.get('sparse') and not isinstance(f_kwargs['sparse'], bool):
                raise TypeError("sparse must be of type bool")
            if f_kwargs.get('copy') and not isinstance(f_kwargs['copy'], bool):
                raise TypeError("copy must be of type bool")
            if f_kwargs.get('mat_ops') and not isinstance(f_kwargs['mat_ops'], dict):
                raise ValueError("mat_ops must be dictionary like.")
            if f_kwargs.get('mld_numeric'):
                if not isinstance(mld_numeric, MldModel):
                    raise TypeError("mld_numeric must be an instance of MldModel")

        return f_kwargs

    @_process_args_mpc_decor
    def _gen_param_struct_tilde(self, N_p=None, param_struct=None, param_struct_subset=None,
                                schedule_params_tilde=None, include_term_cons=True, **kwargs):
        schedule_params_tilde = schedule_params_tilde if schedule_params_tilde is not None else {}
        try:
            schedule_params_tilde.update(kwargs)
        except AttributeError as ae:
            raise TypeError("schedule_params_tilde must be dictionary like or None: " + ae.args[0])

        param_struct_tilde = _copy(param_struct)
        param_struct_tilde['schedule_params_tilde'] = None
        N_cons = N_p + 1 if include_term_cons else N_p
        if schedule_params_tilde:
            param_struct_tilde['schedule_params_tilde'] = [dict.fromkeys(schedule_params_tilde.keys()) for _ in
                                                           range(N_cons)]
            for schedule_param_name, schedule_param_tilde in schedule_params_tilde.items():
                if schedule_param_name not in self._param_struct:
                    raise ValueError(
                        "Invalid schedule_param_name:'{0}' in schedule_params_tilde, name needs to be present in "
                        "param_struct.".format(schedule_param_name))
                elif len(schedule_param_tilde) != (N_cons):
                    raise ValueError(
                        "Invalid length:'{0}' for schedule_param_tilde:'{1}', length of schedule_param_tilde must be "
                        "equal to N_cons:'{2}', where N_cons = N_p+1 if include_term_cons else N_p".format(
                            len(schedule_param_tilde), schedule_param_name, N_cons))
                for k, schedule_param_k in enumerate(schedule_param_tilde):
                    param_struct_tilde['schedule_params_tilde'][k][schedule_param_name] = schedule_param_k

        return param_struct_tilde

    @_process_args_mpc_decor
    def _gen_mld_numeric_tilde(self, N_p=None, param_struct=None, param_struct_subset=None,
                               schedule_params_tilde=None, param_struct_tilde=None, include_term_cons=True,
                               sparse=None, copy=True, **kwargs):
        param_struct_tilde = param_struct_tilde or {}
        try:
            schedule_params_tilde = param_struct_tilde.get('schedule_params_tilde')
        except AttributeError:
            raise TypeError("Invalid type for 'param_struct_tilde', must be dictionary like or None.")

        N_cons = N_p + 1 if include_term_cons else N_p
        if schedule_params_tilde is None:
            mld_numeric_tilde = [self.get_mld_numeric(param_struct_tilde, missing_param_check=False,
                                                      invalid_param_check=False)] * (N_cons)
        elif len(schedule_params_tilde) == N_cons:
            mld_numeric_tilde = [
                self.get_mld_numeric(param_struct=param_struct_tilde, param_struct_subset=schedule_params_tilde[k],
                                     invalid_param_check=False, copy=copy) for k in range(N_cons)
            ]
        else:
            raise ValueError(
                "Invalid length:'{0}' for param_struct_tilde.schedule_param_tilde, length of schedule_param_tilde "
                "must be equal to N_cons:'{1}', where N_cons = N_p+1 if include_term_cons else N_p".format(
                    len(schedule_params_tilde), N_cons))

        return mld_numeric_tilde

    @_process_args_mpc_decor
    def gen_state_input_evolution_matrices(self, N_p=None, param_struct=None, param_struct_subset=None,
                                           schedule_params_tilde=None, param_struct_tilde=None,
                                           include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                                           A_pow_tilde=None, sparse=None, mat_ops=None, copy=True,
                                           **kwargs):

        new_state_input_evolution_struct = StructDict()
        state_gen_kwargs = dict(N_p=N_p,
                                param_struct=param_struct, schedule_params_tilde=schedule_params_tilde,
                                param_struct_tilde=param_struct_tilde, include_term_cons=include_term_cons,
                                mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        new_state_input_evolution_struct['Phi_x'] = self._gen_phi_x(**state_gen_kwargs)
        new_state_input_evolution_struct['Gamma_V'] = self._gen_gamma_V(**state_gen_kwargs)
        new_state_input_evolution_struct['Gamma_W'] = self._gen_gamma_W(**state_gen_kwargs)
        new_state_input_evolution_struct['Gamma_b5'] = self._gen_gamma_b5(**state_gen_kwargs)
        new_state_input_evolution_struct['time_k0'] = self.time_k0

        self.state_input_evolution_struct.update(new_state_input_evolution_struct)

        return new_state_input_evolution_struct

    @_process_args_mpc_decor
    def gen_cons_evolution_matrices(self, N_p=None, param_struct=None, param_struct_subset=None,
                                    schedule_params_tilde=None, param_struct_tilde=None,
                                    include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                                    A_pow_tilde=None, sparse=None, mat_ops=None, copy=True,
                                    **kwargs):

        state_gen_kwargs = dict(N_p=N_p,
                                param_struct=param_struct, schedule_params_tilde=schedule_params_tilde,
                                param_struct_tilde=param_struct_tilde,
                                mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        state_input_evolution_struct = self.gen_state_input_evolution_matrices(**state_gen_kwargs)
        Phi_x = state_input_evolution_struct['Phi_x']
        Gamma_V = state_input_evolution_struct['Gamma_V']
        Gamma_W = state_input_evolution_struct['Gamma_W']
        Gamma_b5 = state_input_evolution_struct['Gamma_b5']

        cons_evo_struct = StructDict()
        con_gen_kwargs = dict(N_p=N_p,
                              param_struct=param_struct, schedule_params_tilde=schedule_params_tilde,
                              param_struct_tilde=param_struct_tilde, include_term_cons=include_term_cons,
                              mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                              A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        E1_tilde = self._gen_E1_tilde_diag(**con_gen_kwargs)
        E_234_tilde = self._gen_E234_tilde_diag(**con_gen_kwargs)
        E5_tilde = self._gen_E5_tilde_diag(**con_gen_kwargs)
        g6_tilde = self._gen_g6_tilde_diag(**con_gen_kwargs)

        cons_evo_struct['H_x'] = E1_tilde @ Phi_x
        cons_evo_struct['H_V'] = E1_tilde @ Gamma_V + E_234_tilde
        cons_evo_struct['H_W'] = -(E1_tilde @ Gamma_W + E5_tilde)
        cons_evo_struct['H_5'] = g6_tilde - E1_tilde @ Gamma_b5

        return cons_evo_struct

    @_process_args_mpc_decor
    def _gen_A_pow_tilde(self, N_p=None, param_struct=None, param_struct_subset=None,
                         schedule_params_tilde=None, param_struct_tilde=None,
                         mld_numeric=None, mld_numeric_tilde=None,
                         sparse=None, mat_ops=None, copy=True,
                         **kwargs):

        # A_pow_tilde = [(A_k)^0, (A_k+1)^1, (A_k+2)^2, ..., (A_k+N_p)^(N_p)]
        if mld_numeric_tilde:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric_tilde[0].A.shape))] + (
                [mat_ops.vmatrix(mld_numeric_tilde[k].A) for k in range(N_p)])
        else:
            A_tilde = [mat_ops.vmatrix(np.eye(*mld_numeric.A.shape))] + [mat_ops.vmatrix(mld_numeric.A)] * (N_p)

        return tuple(itertools.accumulate(A_tilde, lambda x, y: mat_ops.vmatrix(x @ y)))

    @_process_args_mpc_decor
    def _gen_phi_x(self, N_p=None, param_struct=None, param_struct_subset=None,
                   schedule_params_tilde=None, param_struct_tilde=None,
                   mld_numeric=None, mld_numeric_tilde=None,
                   A_pow_tilde=None, sparse=None, mat_ops=None, copy=True,
                   **kwargs):

        # Phi_x = [(A_k)^0; (A_k+1)^1; (A_k+2)^2; ... ;(A_k+N_p)^(N_p)]
        Phi_x = mat_ops.pack.vstack(A_pow_tilde)
        return Phi_x

    @_process_args_mpc_decor
    def _gen_gamma_V(self, N_p=None, param_struct=None, param_struct_subset=None,
                     schedule_params_tilde=None, param_struct_tilde=None,
                     include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                     A_pow_tilde=None, sparse=None, mat_ops=None, copy=True,
                     **kwargs):
        # col = [[0s],(A_k)^0*[B1_k, B2_k, B3_k],..., (A_k+N_p-1)^(N_p-1)*[B1_k+N_p-1, B2_k+N_p-1, B3_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_V = toeplitz(col, row)

        input_mat_names = ['B1', 'B2', 'B3']
        Gamma_V = self._gen_input_evolution_mat(N_p=N_p, include_term_cons=include_term_cons,
                                                mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                                input_mat_names=input_mat_names,
                                                A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        return Gamma_V

    @_process_args_mpc_decor
    def _gen_gamma_W(self, N_p=None, param_struct=None, param_struct_subset=None,
                     schedule_params_tilde=None, param_struct_tilde=None,
                     include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                     A_pow_tilde=None, sparse=None, mat_ops=None, copy=True,
                     **kwargs):

        # col = [[0s],(A_k)^0*[B4],..., (A_k+N_p-1)^(N_p-1)*[B4_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_W = toeplitz(col, row)

        input_mat_names = ['B4']
        Gamma_W = self._gen_input_evolution_mat(N_p=N_p, include_term_cons=include_term_cons,
                                                mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                                input_mat_names=input_mat_names,
                                                A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)
        return Gamma_W

    @_process_args_mpc_decor
    def _gen_gamma_b5(self, N_p=None, param_struct=None, param_struct_subset=None,
                      schedule_params_tilde=None, param_struct_tilde=None,
                      include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                      A_pow_tilde=None, sparse=None, mat_ops=None, copy=True,
                      **kwargs):

        # col = [[0s],(A_k)^0*[b5],..., (A_k+N_p-1)^(N_p-1)*[b5_k+N_p-1]]
        # row = [[0s], ... ,[0s]]
        # Gamma_b5 = toeplitz(col, row)
        input_mat_names = ['b5']
        Gamma_b5 = self._gen_input_evolution_mat(N_p=N_p, include_term_cons=include_term_cons,
                                                 mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                                 input_mat_names=input_mat_names,
                                                 A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops)

        return Gamma_b5

    @_process_args_mpc_decor
    def _gen_E1_tilde_diag(self, N_p=None, param_struct=None, param_struct_subset=None,
                           schedule_params_tilde=None, param_struct_tilde=None,
                           include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                           sparse=None, mat_ops=None, copy=True,
                           **kwargs):

        cons_mat_names = ['E1']

        E1_tilde = self._gen_cons_tilde_diag(N_p=N_p, include_term_cons=include_term_cons,
                                             mld_numeric=mld_numeric, mld_numeric_tilde=mld_numeric_tilde,
                                             cons_mat_names=cons_mat_names,
                                             sparse=sparse, mat_ops=mat_ops)

        return E1_tilde

    @_process_args_mpc_decor
    def _gen_E234_tilde_diag(self, N_p=None, param_struct=None, param_struct_subset=None,
                             schedule_params_tilde=None, param_struct_tilde=None,
                             include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                             sparse=None, mat_ops=None, copy=True,
                             **kwargs):

        cons_mat_names = ['E2', 'E3', 'E4']

        E_234_tilde = self._gen_cons_tilde_diag(N_p=N_p, include_term_cons=include_term_cons, mld_numeric=mld_numeric,
                                                mld_numeric_tilde=mld_numeric_tilde,
                                                cons_mat_names=cons_mat_names,
                                                sparse=sparse, mat_ops=mat_ops)

        return E_234_tilde

    @_process_args_mpc_decor
    def _gen_E5_tilde_diag(self, N_p=None, param_struct=None, param_struct_subset=None,
                           schedule_params_tilde=None, param_struct_tilde=None,
                           include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                           sparse=None, mat_ops=None, copy=True,
                           **kwargs):

        cons_mat_names = ['E5']

        E5_tilde = self._gen_cons_tilde_diag(N_p=N_p, include_term_cons=include_term_cons, mld_numeric=mld_numeric,
                                             mld_numeric_tilde=mld_numeric_tilde,
                                             cons_mat_names=cons_mat_names,
                                             sparse=sparse, mat_ops=mat_ops)

        return E5_tilde

    @_process_args_mpc_decor
    def _gen_g6_tilde_diag(self, N_p=None, param_struct=None, param_struct_subset=None,
                           schedule_params_tilde=None, param_struct_tilde=None,
                           include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                           sparse=None, mat_ops=None, copy=True,
                           **kwargs):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            g6_tilde = np.vstack([mld_numeric_tilde[k].g6 for k in range(N_cons)])
        else:
            g6_tilde = np.vstack([mld_numeric.g6] * (N_cons))

        return g6_tilde

    @_process_args_mpc_decor
    def _test(self, N_p=None, param_struct=None, param_struct_subset=None,
              schedule_params_tilde=None, param_struct_tilde=None,
              include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
              A_pow_tilde=None, sparse=None, mat_ops=None, copy=True,
              **kwargs):
        # print(N_p)
        # print(mld_numeric)

        return 1

    @_process_args_mpc_decor
    def _test2(self):
        return 1

    @staticmethod
    def _gen_input_evolution_mat(N_p=None, include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                                 input_mat_names=None, A_pow_tilde=None, sparse=False, mat_ops=None):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            B_hstack_tilde = [
                mat_ops.vmatrix(mat_ops.pack.hstack(
                    [mat_ops.hmatrix(mld_numeric_tilde[k][input_mat_name]) for input_mat_name in input_mat_names])
                ) for k in range(N_p)]

            col_list = ([mat_ops.zeros(B_hstack_tilde[0].shape)] +
                        [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack_tilde[k]) for k in range(N_p)])
            row_list = [mat_ops.zeros(B_hstack_tilde[0].shape)] * (N_cons)
        else:
            B_hstack = mat_ops.vmatrix(mat_ops.pack.hstack(
                [mat_ops.hmatrix(mld_numeric[input_mat_name]) for input_mat_name in input_mat_names]))

            col_list = ([mat_ops.zeros(B_hstack.shape)] +
                        [mat_ops.vmatrix(A_pow_tilde[k] @ B_hstack) for k in range(N_p)])
            row_list = [mat_ops.zeros(B_hstack.shape)] * (N_cons)

        return _block_toeplitz(col_list, row_list, sparse=sparse)

    @staticmethod
    def _gen_cons_tilde_diag(N_p=None, include_term_cons=True, mld_numeric=None, mld_numeric_tilde=None,
                             cons_mat_names=None, sparse=False, mat_ops=None):

        N_cons = N_p + 1 if include_term_cons else N_p
        if mld_numeric_tilde:
            E_hstack_tilde = [
                mat_ops.vmatrix(mat_ops.pack.hstack(
                    [mat_ops.hmatrix(mld_numeric_tilde[k][cons_mat_name]) for cons_mat_name in cons_mat_names])
                ) for k in range(N_cons)]

            E_hstack_tilde_diag = mat_ops.block_diag(E_hstack_tilde)
        else:
            E_hstack_tilde = [mat_ops.vmatrix(mat_ops.pack.hstack(
                [mat_ops.hmatrix(mld_numeric[cons_mat_name]) for cons_mat_name in cons_mat_names]))] * N_cons

            E_hstack_tilde_diag = mat_ops.block_diag(E_hstack_tilde)

        # if is_input_cons:
        #     E_tilde_diag = mat_ops.pack.vstack(
        #         [E_hstack_tilde_diag, mat_ops.zeros((E_hstack_tilde[0].shape[0], E_hstack_tilde[0].shape[1] *
        # N_cons))])
        # else:
        #     E_tilde_diag = E_hstack_tilde_diag

        return E_hstack_tilde_diag

    @staticmethod
    @_cache_hashable_args(maxsize=2)
    def _get_mat_ops(sparse=False):
        mat_ops = StructDict()
        if sparse:
            mat_ops['pack'] = scs
            mat_ops['linalg'] = scs
            mat_ops['sclinalg'] = scs
            mat_ops['block_diag'] = scs.block_diag
            mat_ops['vmatrix'] = scs.csr_matrix
            mat_ops['hmatrix'] = scs.csc_matrix
            mat_ops['zeros'] = scs.csr_matrix
        else:
            mat_ops['pack'] = np
            mat_ops['linalg'] = np.linalg
            mat_ops['sclinalg'] = scl
            mat_ops['block_diag'] = _block_diag_dense
            mat_ops['vmatrix'] = np.atleast_2d
            mat_ops['hmatrix'] = np.atleast_2d
            mat_ops['zeros'] = np.zeros
        return mat_ops


def _to_dense(arr):
    pass


def _block_diag_dense(mats, format=None, dtype=None):
    block_diag = scl.block_diag(*mats)
    if dtype is not None:
        block_diag = block_diag.astype(dtype)
    return block_diag


def _create_object_array(tup):
    try:
        obj_arr = np.empty(len(tup), dtype='object')
    except TypeError:
        raise TypeError("tup must be array like.")

    for ind, item in enumerate(tup):
        obj_arr[ind] = item
    return obj_arr


def _atleast_3d_toeplitz(arr):
    # used to ensure block toeplitz is compatible with scipy.linalg.toeplitz
    arr = np.array(arr)
    ndim = arr.ndim
    if ndim < 3:
        return np.moveaxis(np.atleast_3d(arr), ndim, 0)
    else:
        return np.atleast_3d(arr)


def _block_toeplitz(c_tup, r_tup=None, sparse=False):
    """
     Based on scipy.linalg.toeplitz method but applied in a block fashion.
     """
    try:
        c = np.array(c_tup)
    except ValueError:
        c = _create_object_array(c_tup)

    if r_tup is None:
        if np.issubdtype(c.dtype, np.number):
            r = c.conjugate()
        else:
            r = c
    else:
        try:
            r = np.array(r_tup)
        except ValueError:
            r = _create_object_array(r_tup)

    c = _atleast_3d_toeplitz(c)
    r = _atleast_3d_toeplitz(r)
    # # Form a array containing a reversed c followed by r[1:] that could be strided to give us a toeplitz matrix.
    try:
        vals = np.concatenate((c[::-1], r[1:]))
    except ValueError as ve:
        raise ValueError("Incompatible dimensions in c_tup or between c_tup and r_tup - " + ve.args[0])
    stride_shp = (c.shape[0], c.shape[1], r.shape[0], r.shape[2])
    out_shp = (c.shape[0] * c.shape[1], r.shape[0] * r.shape[2])
    n, m, k = vals.strides

    np_toep_strided = _as_strided(vals[c.shape[0] - 1:], shape=stride_shp, strides=(-n, m, n, k)).reshape(out_shp)

    if sparse:
        if np_toep_strided.dtype != np.object_:
            return scs.csr_matrix(np_toep_strided)
        elif all(isinstance(block, scs.csr_matrix) for block in np_toep_strided.flat):
            v_stacked = [scs.bmat(np.atleast_2d(col).T).tocsc() for col in np_toep_strided.T]
            return scs.bmat(np.atleast_2d(v_stacked)).tocsr()
        else:
            h_stacked = [scs.bmat(np.atleast_2d(row)).tocsr() for row in np_toep_strided]
            return scs.bmat(np.atleast_2d(h_stacked).T).tocsc()
    else:
        return np_toep_strided.copy()


def _block_toeplitz_alt(c_tup, r_tup=None, sparse=False):
    c = _create_object_array(c_tup)
    if r_tup is None:
        try:
            r = c.conjugate()
        except AttributeError:
            r = c
    else:
        r = _create_object_array(r_tup)
    # # Form a 1D array containing a reversed c followed by r[1:] that could be
    # # strided to give us toeplitz matrix.
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = c.shape[0], r.shape[0]
    n = vals.strides[0]
    strided = _as_strided(vals[len(c) - 1:], shape=out_shp, strides=(-n, n))
    np_toep = np.block(strided.tolist())

    if sparse:
        if all(isinstance(block, scs.csr_matrix) for block in np_toep.flat):
            v_stacked = [scs.bmat(np.atleast_2d(col).T).tocsc() for col in np_toep.T]
            return scs.bmat(np.atleast_2d(v_stacked)).tocsr()
        else:
            h_stacked = [scs.bmat(np.atleast_2d(row)).tocsr() for row in np_toep]
            return scs.bmat(np.atleast_2d(h_stacked).T).tocsc()
    else:
        return np_toep


class AgentRepository:
    pass
