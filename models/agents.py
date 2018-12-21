import bisect
from collections import OrderedDict
from copy import copy as _copy
from reprlib import recursive_repr as _recursive_repr

# import pandas as pd
# pd.set_option('mode.chained_assignment', 'raise')

from models.mld_model import MldModel
from tools.mpc_tools import MpcEvoGenerator, MpcOptVariables
from utils.decorator_utils import ParNotReq, process_method_args_decor
from utils.structdict import StructDict, struct_repr

from utils.helper_funcs import num_not_None


class AgentModel:
    MldNames = MldModel.MldModelTypesNamedTup(numeric='mld_numeric', callable='mld_callable', symbolic='mld_symbolic')

    def __init__(self, mld_numeric=None, mld_callable=None, mld_symbolic=None, param_struct=None, copy=True):
        self._param_struct = None
        self._mld_numeric = None
        self._mld_callable = None
        self._mld_symbolic = None
        self.update_mld(mld_numeric=mld_numeric, mld_callable=mld_callable, mld_symbolic=mld_symbolic,
                        param_struct=param_struct, copy=copy, missing_param_check=True)

    def update_mld(self, mld_numeric=None, mld_callable=None, mld_symbolic=None, param_struct=None,
                   param_struct_subset=None, copy=True, missing_param_check=True, invalid_param_check=False, **kwargs):

        mlds = MldModel.MldModelTypesNamedTup(numeric=mld_numeric, callable=mld_callable, symbolic=mld_symbolic)
        if num_not_None(mlds) > 1:
            raise ValueError(
                f"Only one of {{'mld_numeric', 'mld_callable', 'mld_symbolic'}} can be used to construct/update an "
                f"{self.__class__.__name__}")
        elif not all(isinstance(mld, MldModel) or mld is None for mld in mlds):
            raise TypeError(f"Each of {{'mld_numeric', 'mld_callable', 'mld_symbolic'}} is required to be an "
                            f"instance of {MldModel.__name__} or None.")
        for index, mld in enumerate(mlds):
            if mld is not None and mld.mld_type != MldModel.MldModelTypes[index]:
                raise TypeError(
                    f"'{self.MldNames[index]}' is required to be an instance of {MldModel.__name__} with mld_type:"
                    f"{MldModel.MldModelTypes[index]}, not mld_type:'{mld.mld_type}'"
                )

        if any(mlds):
            self._mld_numeric = mlds.numeric
            self._mld_callable = mlds.callable
            self._mld_symbolic = mlds.symbolic

        param_struct = param_struct if param_struct is not None else (self._param_struct or {})
        try:
            self._param_struct = self._validate_param_struct(param_struct=param_struct,
                                                             param_struct_subset=param_struct_subset,
                                                             missing_param_check=missing_param_check,
                                                             invalid_param_check=invalid_param_check,
                                                             **kwargs)
        except ValueError as ve:
            raise ValueError(
                f"A valid 'param_struct' is required, the argument was not provided or is invalid. {ve.args[0]}")

        if self._mld_callable or self._mld_symbolic:
            self._mld_callable = self._mld_callable or self._mld_symbolic.to_callable(copy=copy)
            self._mld_numeric = self._mld_callable.to_numeric(param_struct=param_struct, copy=copy)

    @property
    def mld_numeric(self):
        return _copy(self._mld_numeric)

    @property
    def mld_callable(self):
        return _copy(self._mld_callable)

    @property
    def mld_symbolic(self):
        return _copy(self._mld_symbolic)

    @property
    def param_struct(self):
        return _copy(self._param_struct)

    @param_struct.setter
    def param_struct(self, param_struct):
        self.update_param_struct(param_struct=param_struct)

    def update_param_struct(self, param_struct=None, param_struct_subset=None, missing_param_check=True,
                            invalid_param_check=False, **kwargs):
        param_struct = self._validate_param_struct(param_struct=param_struct,
                                                   param_struct_subset=param_struct_subset,
                                                   missing_param_check=missing_param_check,
                                                   invalid_param_check=invalid_param_check, **kwargs)

        self._mld_numeric = self.get_mld_numeric(param_struct=param_struct, _bypass_param_struct_validation=True)
        self._param_struct = param_struct

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
            param_struct = _copy(self._param_struct)
            given_params = param_struct_subset
            param_struct.update(param_struct_subset)
        else:
            param_struct = _copy(param_struct)
            try:
                param_struct.update(param_struct_subset)
            except AttributeError:
                raise TypeError("Invalid type for 'param_struct', must be dictionary like or None.")
            given_params = param_struct

        if missing_param_check:
            required_params = set(self.get_required_params())
            if required_params:
                missing_keys = required_params.difference(param_struct.keys())
                if missing_keys:
                    raise ValueError(
                        f"The following keys are missing from param_struct: '{missing_keys}'"
                    )

        if invalid_param_check:
            invalid_params = set(given_params.keys()).difference(self._param_struct.keys())
            if invalid_params:
                raise ValueError(
                    f"Invalid keys:'{invalid_params}' in kwargs/param_struct - keys must all exist in "
                    f"self.param_struct. Hint: either disable 'invalid_param_check' or update self.param_struct."
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
                return self._mld_callable.to_numeric(compute_param_struct, copy=copy)
            except AttributeError:
                raise TypeError("AgentModel does not contain valid mld_callable.")
        else:
            return self._mld_numeric

    def get_required_params(self):
        if self._mld_symbolic:
            return set(self._mld_symbolic.mld_info.required_params)
        elif self._mld_callable:
            return set(self._mld_callable.mld_info.required_params)
        else:
            return set()

    @_recursive_repr()
    def __repr__(self):
        repr_dict = OrderedDict(mld_numeric=self._mld_numeric,
                                mld_callable=self._mld_callable,
                                mld_symbolic=self._mld_symbolic)
        return struct_repr(repr_dict, type_name=self.__class__.__name__)


class PvAgentModel(AgentModel):

    def _process_param_struct_args(self, f_kwargs=None,
                                   param_struct=ParNotReq, param_struct_subset=None):

        _param_struct = self._validate_param_struct(
            param_struct=f_kwargs.get('param_struct'), param_struct_subset=param_struct_subset,
            missing_param_check=False, invalid_param_check=False)

        if param_struct is None:
            f_kwargs['param_struct'] = _param_struct

        return f_kwargs

    @process_method_args_decor('_process_param_struct_args')
    def _gen_schedule_params_tilde(self, N_tilde, param_struct=None, param_struct_subset=None,
                                   schedule_params_evo=None, **kwargs):

        schedule_params_evo = schedule_params_evo if schedule_params_evo is not None else {}
        try:
            schedule_params_evo.update({key: value for key, value in kwargs.items() if key in param_struct})
        except AttributeError as ae:
            raise TypeError("schedule_params_evo must be dictionary like or None: " + ae.args[0])

        if schedule_params_evo:
            schedule_params_tilde = [dict.fromkeys(schedule_params_evo.keys()) for _ in range(N_tilde)]
            for schedule_param_name, schedule_param_tilde in schedule_params_evo.items():
                if schedule_param_name not in self._param_struct:
                    raise ValueError(
                        "Invalid schedule_param_name:'{0}' in schedule_params_evo, name needs to be present in "
                        "param_struct.".format(schedule_param_name))
                elif len(schedule_param_tilde) != (N_tilde):
                    raise ValueError(
                        "Invalid length:'{0}' for schedule_param_tilde:'{1}', length of schedule_param_tilde must be "
                        "equal to N_tilde:'{2}'".format(
                            len(schedule_param_tilde), schedule_param_name, N_tilde))
                for k, schedule_param_k in enumerate(schedule_param_tilde):
                    schedule_params_tilde[k][schedule_param_name] = schedule_param_k
        else:
            schedule_params_tilde = None

        return schedule_params_tilde

    @process_method_args_decor('_process_param_struct_args')
    def get_mld_numeric_tilde(self, N_tilde, param_struct=None, param_struct_subset=None,
                              schedule_params_tilde=None, copy=None, **kwargs):

        if schedule_params_tilde is None:
            mld_numeric_tilde = [self.get_mld_numeric(param_struct, missing_param_check=False,
                                                      invalid_param_check=False, copy=copy)] * (N_tilde)
        elif len(schedule_params_tilde) == N_tilde:
            mld_numeric_tilde = [
                self.get_mld_numeric(param_struct=param_struct, param_struct_subset=schedule_params_tilde[k],
                                     invalid_param_check=False, copy=copy) for k in range(N_tilde)
            ]
        else:
            raise ValueError(
                "Invalid length:'{0}' for param_struct_tilde.schedule_param_tilde, length of schedule_param_tilde "
                "must be equal to N_tilde:'{1}'".format(len(schedule_params_tilde), N_tilde))

        return mld_numeric_tilde


class Agent:
    _agent_type_id_struct = StructDict()

    def __init__(self, agent_type=None, agent_id=None, agent_model=None):

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

    def __init__(self, agent_type=None, agent_id=None, agent_model=None, N_p=None, include_term_cons=True):
        super().__init__(agent_type=agent_type, agent_id=agent_id, agent_model=agent_model)
        self._N_p = N_p if N_p is not None else 0
        self._include_term_cons = include_term_cons

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
        return None

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
