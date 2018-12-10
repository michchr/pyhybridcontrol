import bisect
from collections import OrderedDict
from copy import copy as _copy
from reprlib import recursive_repr as _recursive_repr

import pandas as pd

from models.mld_model import MldModel
from tools.mpc_tools import MpcEvoGenerator, MpcOptVariables
from utils.decorator_utils import ParNotReq, process_method_args_decor
from utils.structdict import StructDict, struct_repr

pd.set_option('mode.chained_assignment', 'raise')


class AgentModelGenerator:

    def __init__(self, *args, **kwargs):
        self.mld_symbolic = self.get_mld_symbolic(*args, **kwargs)
        self.mld_eval = self.get_mld_eval()

    def get_mld_symbolic(self, *args, **kwargs):
        raise NotImplementedError("Need to implement symbolic mld in subclass.")

    def get_mld_eval(self, mld_symbolic=None):
        mld_symbolic = mld_symbolic or self.mld_symbolic
        return mld_symbolic.to_eval()

    def get_mld_numeric(self, param_struct=None, mld_model=None, copy=True):
        mld_model = mld_model or self.mld_eval or self.mld_symbolic
        return mld_model.to_numeric(param_struct=param_struct, copy=copy)

    def get_required_params(self):
        return self.mld_symbolic.mld_info['required_params']


class Agent:
    _agent_type_id_struct = StructDict()

    def __init__(self, agent_type=None, agent_id=None, agent_model_generator = None,
                 param_struct=None, mld_numeric = None):

        self._mld_numeric = mld_numeric
        self._agent_model_generator = agent_model_generator
        self._param_struct = {}
        self._param_struct = self._validate_param_struct(param_struct=param_struct, missing_param_check=True,
                                                         invalid_param_check=False)
        if self._agent_model_generator is not None:
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
        try:
            self.update_mld_numeric(param_struct=param_struct, param_struct_subset=param_struct_subset,
                                    missing_param_check=missing_param_check,
                                    invalid_param_check=invalid_param_check, **kwargs)
        except TypeError:
            pass
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
        elif param_struct is self._param_struct:
            given_params = param_struct_subset
            param_struct = _copy(self._param_struct)
            param_struct.update(param_struct_subset)
        else:
            param_struct = _copy(param_struct)
            try:
                param_struct.update(param_struct_subset)
            except AttributeError:
                raise TypeError("Invalid type for 'param_struct', must be dictionary like or None.")
            given_params = param_struct

        if missing_param_check:
            try:
                required_params = set(self._agent_model_generator.get_required_params())
            except TypeError:
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

        if kwargs.pop('_bypass_param_struct_validation', False):
            compute_param_struct = param_struct
        else:
            compute_param_struct = self._validate_param_struct(param_struct=param_struct,
                                                               param_struct_subset=param_struct_subset,
                                                               missing_param_check=missing_param_check,
                                                               invalid_param_check=invalid_param_check,
                                                               **kwargs)

        if compute_param_struct is not self._param_struct:
            try:
                return self._agent_model_generator.get_mld_numeric(compute_param_struct, copy=copy)
            except AttributeError:
                raise TypeError("Agent does not contain valid agent_model_generator.")
        else:
            return self._mld_numeric

    def update_mld_numeric(self, param_struct=None, param_struct_subset=None, missing_param_check=False,
                        invalid_param_check=False, mld_numeric=None, copy=True, **kwargs):
        if isinstance(mld_numeric, MldModel) and mld_numeric.mld_type == mld_numeric.MldModelTypes.numeric:
            self._mld_numeric = mld_numeric
        else:
            new_param_struct = self._validate_param_struct(param_struct=param_struct,
                                                           param_struct_subset=param_struct_subset,
                                                           missing_param_check=missing_param_check,
                                                           invalid_param_check=invalid_param_check,
                                                           **kwargs)
            self._mld_numeric = self.get_mld_numeric(param_struct=new_param_struct, copy=copy,
                                                     _bypass_param_struct_validation=True)
            self._param_struct = new_param_struct

        return self.mld_numeric

    @_recursive_repr()
    def __repr__(self):
        repr_dict = OrderedDict(agent_type=self.agent_type, agent_id=self.agent_id,
                                agent_model_generator=self._agent_model_generator,
                                mld_numeric=self._mld_numeric)
        return struct_repr(repr_dict, type_name=self.__class__.__name__)


class MpcAgent(Agent):

    def __init__(self, *args, N_p=None, include_term_cons=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._N_p = N_p if N_p is not None else 0
        self._include_term_cons = include_term_cons
        self._mld_numeric_tilde = None
        self._mpc_evo_gen = MpcEvoGenerator(self)
        self._opt_var_gen = MpcOptVariables(self)

    @property
    def N_p(self):
        return self._N_p

    @property
    def include_term_cons(self):
        return self._include_term_cons

    @property
    def mld_numeric_tilde(self):
        return self._mld_numeric_tilde

    def get_mpc_evo_struct(self, N_p=None, sparse=None):
        return self._mpc_evo_gen.gen_mpc_evolution_matrices(N_p=N_p, sparse=sparse)

    def get_cur_objective(self, q_U_N_p=None, q_Delta_N_p=None, q_Z_N_p=None, q_X_N_p=None,
                          q_X_F=None):
        pass
        # if mld_numeric_tilde:
        #     mld_info = mld_numeric_tilde[0].mld_info
        # else:
        #     mld_info = mld_numeric.mld_info
        #
        # gen_kwargs = dict(_disable_process_args=True, N_p=N_p, param_struct=param_struct,
        #                   param_struct_subset=param_struct_subset, param_struct_tilde=param_struct_tilde,
        #                   schedule_params_tilde=schedule_params_tilde,
        #                   include_term_cons=include_term_cons, mld_numeric=mld_numeric,
        #                   mld_numeric_tilde=mld_numeric_tilde,
        #                   A_pow_tilde=A_pow_tilde, sparse=sparse, mat_ops=mat_ops, copy=copy,
        #                   **kwargs)
        #
        # opt_vars = self._gen_optimization_vars(**gen_kwargs)
        # state_input_mat_evo = self.gen_state_input_evolution_matrices(**gen_kwargs)
        #
        # X_tilde_N_cons = (state_input_mat_evo['Phi_x'] @ np.array([[0]]) + state_input_mat_evo['Gamma_V'] @
        #                   opt_vars['V_tilde_N_cons'] + state_input_mat_evo['Gamma_W'][:, :1] * 0 +
        #                   state_input_mat_evo['Gamma_b5'])
        # X_tilde_N_p = X_tilde_N_cons[:(N_p * mld_info['nx']), :]
        #
        # obj = cvx.Constant(0)
        # if q_U_N_p is not None:
        #     obj += np.transpose(q_U_N_p) @ opt_vars['U_tilde_N_p']
        # if q_Delta_N_p is not None:
        #     obj += np.transpose(q_Delta_N_p) @ opt_vars['Delta_tilde_N_p']
        # if q_Z_N_p is not None:
        #     obj += np.transpose(q_Delta_N_p) @ opt_vars['Z_tilde_N_p']
        # if q_X_N_p is not None:
        #     obj += np.transpose(q_X_N_p) @ X_tilde_N_p
        # if q_X_F is not None and include_term_cons:
        #     obj += np.transpose(q_X_F) @ X_tilde_N_cons[-(mld_info['nx']):, :]
        #
        # return obj

    def _process_base_args(self, f_kwargs=None, *,
                           N_p=ParNotReq, include_term_cons=ParNotReq, copy=ParNotReq):

        if N_p is None:
            f_kwargs['N_p'] = self.N_p

        if include_term_cons is None:
            f_kwargs['include_term_cons'] = self.include_term_cons

        if copy is None:
            f_kwargs['copy'] = True

        return f_kwargs

    def _process_param_struct_args(self, f_kwargs=None,
                                   N_p=None, include_term_cons=None,
                                   param_struct=ParNotReq, param_struct_subset=None,
                                   schedule_params_tilde=ParNotReq, schedule_params_evo=None,
                                   **kwargs):

        _param_struct = self._validate_param_struct(
            param_struct=f_kwargs.get('param_struct'), param_struct_subset=param_struct_subset,
            missing_param_check=False, invalid_param_check=False)

        if param_struct is None:
            f_kwargs['param_struct'] = _param_struct

        if schedule_params_evo or kwargs and schedule_params_tilde is None:
            f_kwargs['schedule_params_tilde'] = (
                self._gen_schedule_params_tilde(_disable_process_args=True,
                                                N_p=N_p, include_term_cons=include_term_cons,
                                                param_struct=_param_struct,
                                                schedule_params_evo=schedule_params_evo,
                                                **kwargs))

        return f_kwargs

    def _process_mld_model_args(self, f_kwargs=None, *, mld_numeric=ParNotReq, mld_numeric_tilde=ParNotReq):
        if f_kwargs.get('param_struct') is not None:
            if f_kwargs.get('param_struct') is self._param_struct:
                _mld_numeric = self.mld_numeric
            else:
                _mld_numeric = self.get_mld_numeric(param_struct=f_kwargs.get('param_struct'),
                                                    missing_param_check=False,
                                                    invalid_param_check=False, copy=f_kwargs.get('copy'))
        else:
            _mld_numeric = self.mld_numeric

        if f_kwargs.get('param_struct_tilde') is not None and mld_numeric_tilde is None:
            f_kwargs['mld_numeric_tilde'] = (
                self._gen_mld_numeric_tilde(_disable_process_args=True,
                                            N_p=f_kwargs['N_p'], include_term_cons=f_kwargs['include_term_cons'],
                                            param_struct=f_kwargs['param_struct'],
                                            param_struct_tilde=f_kwargs['param_struct_tilde'],
                                            copy=f_kwargs['copy']))

        if mld_numeric is None:
            f_kwargs['mld_numeric'] = _mld_numeric

        return f_kwargs

    @process_method_args_decor('_process_base_args', '_process_param_struct_args')
    def _gen_schedule_params_tilde(self, N_p=None, include_term_cons=None, param_struct=None,
                                   param_struct_subset=None, schedule_params_evo=None, **kwargs):

        schedule_params_evo = schedule_params_evo if schedule_params_evo is not None else {}
        try:
            schedule_params_evo.update({key: value for key, value in kwargs.items() if key in param_struct})
        except AttributeError as ae:
            raise TypeError("schedule_params_evo must be dictionary like or None: " + ae.args[0])

        N_cons = N_p + 1 if include_term_cons else N_p
        if schedule_params_evo:
            schedule_params_tilde = [dict.fromkeys(schedule_params_evo.keys()) for _ in range(N_cons)]
            for schedule_param_name, schedule_param_tilde in schedule_params_evo.items():
                if schedule_param_name not in self._param_struct:
                    raise ValueError(
                        "Invalid schedule_param_name:'{0}' in schedule_params_evo, name needs to be present in "
                        "param_struct.".format(schedule_param_name))
                elif len(schedule_param_tilde) != (N_cons):
                    raise ValueError(
                        "Invalid length:'{0}' for schedule_param_tilde:'{1}', length of schedule_param_tilde must be "
                        "equal to N_cons:'{2}', where N_cons = N_p+1 if include_term_cons else N_p".format(
                            len(schedule_param_tilde), schedule_param_name, N_cons))
                for k, schedule_param_k in enumerate(schedule_param_tilde):
                    schedule_params_tilde[k][schedule_param_name] = schedule_param_k
        else:
            schedule_params_tilde = None

        return schedule_params_tilde

    @process_method_args_decor('_process_base_args', '_process_param_struct_args', '_process_mld_model_args')
    def _gen_mld_numeric_tilde(self, N_p=None, include_term_cons=None, param_struct=None,
                               param_struct_subset=None, schedule_params_tilde=None,
                               copy=None, **kwargs):

        N_cons = N_p + 1 if include_term_cons else N_p
        if schedule_params_tilde is None:
            mld_numeric_tilde = [self.get_mld_numeric(param_struct, missing_param_check=False,
                                                      invalid_param_check=False)] * (N_cons)
        elif len(schedule_params_tilde) == N_cons:
            mld_numeric_tilde = [
                self.get_mld_numeric(param_struct=param_struct, param_struct_subset=schedule_params_tilde[k],
                                     invalid_param_check=False, copy=copy) for k in range(N_cons)
            ]
        else:
            raise ValueError(
                "Invalid length:'{0}' for param_struct_tilde.schedule_param_tilde, length of schedule_param_tilde "
                "must be equal to N_cons:'{1}', where N_cons = N_p+1 if include_term_cons else N_p.".format(
                    len(schedule_params_tilde), N_cons))

        return mld_numeric_tilde


class AgentRepository:
    pass
