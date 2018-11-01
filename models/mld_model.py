import numpy as np
import sympy as sp
import scipy.sparse as scs
import scipy.linalg as scl
import scipy.interpolate as sci
import functools
from enum import Enum
from copy import deepcopy as _deepcopy
from collections import namedtuple as NamedTuple
import inspect
from sortedcontainers import SortedDict

from utils.structdict import StructDict, SortedStructDict, OrderedStructDict

from datetime import (
    datetime as DateTime,
    timedelta as TimeDelta)

from pandas import date_range


def _process_args_decor(update_info_structs=False):
    def decor_func(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            kwonlyargs_names = inspect.getfullargspec(func).kwonlyargs
            if args and (args[0] is None or callable(args[0])) and isinstance(self, SortedDict):
                self._key = args[0]
                args = args[1:]

            if kwonlyargs_names:
                try:
                    kwargs.update({kwonlyargs_names[i]: arg for (i, arg) in enumerate(args)})
                except IndexError:
                    raise TypeError("Too many args supplied!")

            if update_info_structs:
                kwargs = MldInfo._update_info_structs_from_kwargs(kwargs=kwargs)

            return func(self, *args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decor_func


class MldBase(SortedStructDict):
    __internal_names = []
    _internal_names_set = SortedStructDict._internal_names_set.union(__internal_names)
    _allowed_data_set = set()

    def _init_std_attributes(self, *args, **kwargs):
        _sdict = super(MldBase, self)
        _sdict._init_std_attributes(*args, **kwargs)
        self._sdict_init = _sdict.__init__
        self._sdict_setitem = _sdict.__setitem__
        self._sdict_update = _sdict.update

    # noinspection PyMissingConstructor
    @_process_args_decor()
    def __init__(self, *args, **kwargs):
        if hasattr(self, '_key'):
            # noinspection PyUnresolvedReferences
            self._sdict_init(self._key)
        else:
            self._sdict_init()
        self._sdict_update(dict.fromkeys(self._allowed_data_set))

    def __setitem__(self, key, value):
        if key in self.keys():
            self.update(**{key: value})
        elif key in self._internal_names_set:
            object.__setattr__(self, key, value)
        else:
            raise KeyError("key:'{}' is not valid or does not exist".format(key))

    def __copy__(self):
        return self.__class__(self)


_mld_types = ['numeric', 'symbolic', 'callable']
MldType = Enum('MldType', _mld_types)

_MldMatType = NamedTuple('_MldMatType', ['state_input', 'output', 'constraint'])

_mld_dim_map = {
    # Max column dimension of (*):
    'nx': ('A', 'C', 'E1'),
    'nu': ('B1', 'D1', 'E2'),
    'ndelta': ('B2', 'D2', 'E3'),
    'nz': ('B3', 'D3', 'E4'),
    'nomega': ('B4', 'D4', 'E5'),

    # Max row dimension of (*):
    'n_states': ('A', 'B1', 'B2', 'B3', 'B4', 'b5'),
    'n_outputs': ('C', 'D1', 'D2', 'D3', 'D4', 'd5'),
    'n_cons': ('E1', 'E2')
}


class MldModel(MldBase):
    __internal_names = ['mld_info', 'mld_type']
    _internal_names_set = MldBase._internal_names_set.union(__internal_names)

    _state_input_matrix_names = ['A', 'B1', 'B2', 'B3', 'B4', 'b5']
    _out_matrix_names = ['C', 'D1', 'D2', 'D3', 'D4', 'd5']
    _con_matrix_names = ['E1', 'E2', 'E3', 'E4', 'E5', 'g6']

    _sys_matrix_names = _MldMatType(_state_input_matrix_names, _out_matrix_names, _con_matrix_names)
    _offset_vect_names = _MldMatType(_state_input_matrix_names[-1], _out_matrix_names[-1], _con_matrix_names[-1])
    _allowed_data_set = set([name for names in _sys_matrix_names for name in names])

    @_process_args_decor()
    def __init__(self, *args, system_matrices=None, dt=None, bin_dims_struct=None, var_types_struct=None, **kwargs):

        super(MldModel, self).__init__(*args, **kwargs)

        self._sdict_update(dict.fromkeys(self._allowed_data_set, np.empty(shape=(0, 0))))
        self.mld_info = None
        self.mld_type = None

        self.update(*args[:4], system_matrices=system_matrices, dt=dt, bin_dims_struct=bin_dims_struct,
                    var_types_struct=var_types_struct, from_init=True, **kwargs)

    def __reduce__(self):
        if hasattr(self, '_key'):
            args = (self._key, list(self.items()), None, self.mld_info.var_types_struct)
        else:
            args = (list(self.items()), None, self.mld_info.var_types_struct)
        return (self.__class__, args)

    @_process_args_decor(update_info_structs=True)
    def update(self, *args, system_matrices=None, dt=None, bin_dims_struct=None, var_types_struct=None, **kwargs):
        from_init = kwargs.pop('from_init', None)
        if system_matrices and kwargs:
            raise ValueError("Individual matrix arguments cannot be set if 'system_matrices' argument is set")

        if args and isinstance(args[0], self.__class__):
            self.mld_info = args[0].mld_info

        if not isinstance(self.mld_info, MldInfo):
            self.mld_info = MldInfo()

        creation_matrices = system_matrices or kwargs
        if not isinstance(creation_matrices, dict):
            try:
                creation_matrices = dict(creation_matrices)
            except Exception:
                pass

        new_sys_mats = StructDict(self.items())
        try:
            for sys_matrix_id, system_matrix in creation_matrices.items():
                if sys_matrix_id in self._allowed_data_set:
                    old_val = self.get(sys_matrix_id)
                    if isinstance(system_matrix, (sp.Expr)):
                        system_matrix = sp.Matrix([system_matrix])
                    elif not isinstance(system_matrix, sp.Matrix) and not callable(system_matrix):
                        system_matrix = np.atleast_2d(system_matrix)
                        # if not np.prod(system_matrix.shape):
                        #     system_matrix = system_matrix.reshape(0, 0)
                    new_sys_mats[sys_matrix_id] = system_matrix if system_matrix is not None else old_val
                else:
                    if kwargs:
                        raise ValueError("Invalid matrix name in kwargs: {}".format(sys_matrix_id))
                    else:
                        raise ValueError("Invalid matrix name in system_matrices: {}".format(sys_matrix_id))
        except AttributeError:
            raise TypeError("Argument:'system_matrices' must be dictionary like")

        if creation_matrices.get('C') is None and not callable(new_sys_mats['C']) and from_init:
            if isinstance(new_sys_mats['A'], sp.Matrix):
                new_sys_mats['C'] = sp.Matrix(np.eye(*new_sys_mats['A'].shape))
            else:
                new_sys_mats['C'] = np.eye(*new_sys_mats['A'].shape)

        try:
            new_sys_mats = self._validate_sys_matrix_shapes(new_sys_mats)
            self._set_mld_type(new_sys_mats)
        except ValueError as ve:
            raise ve

        self._sdict_update(new_sys_mats)
        self.mld_info.update(self, dt=dt, bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct)

    def lsim(self, u=None, delta=None, z=None, omega=None, x0=None, t=None, start_datetime=None, end_datetime=None):
        f_locals = locals()
        # create struct_dict of all input sequences
        inputs_struct = StructDict(
            {input_name: _as2darray(f_locals.get(input_name)) for input_name in self.mld_info._input_names})

        max_input_samples = _get_expr_shapes(inputs_struct, get_max_dim=True)[0]
        if t is None:
            out_samples = max_input_samples
            stoptime = (out_samples - 1) * self.mld_info.dt
        else:
            stoptime = t[-1]
            out_samples = int(np.floor(stoptime / self.mld_info.dt)) + 1

        for input_name, input in inputs_struct.items():
            req_input_dim_name = ''.join(['n', input_name])
            req_input_dim = self.mld_info.get(req_input_dim_name)
            if req_input_dim == 0:
                if input.shape[1] != 0:
                    raise ValueError(
                        "Invalid shape for input sequence'{0}':{1}, column dimension must be equal to 0 as required "
                        "input is null, i.e. '(*,{2})')".format(input_name, input.shape, 0)
                    )
                elif input.shape[0] != out_samples:  # ensure null inputs have correct dimension
                    inputs_struct[input_name] = inputs_struct[input_name].reshape(out_samples, 0)
            elif input.shape[1] == req_input_dim:
                if t is None and input.shape[0] != max_input_samples:
                    raise ValueError(
                        "Invalid shape for input sequence'{0}':{1}, row dimension must be equal to maximum row "
                        "dimension of all input sequences, i.e. '({2},*)')".format(
                            input_name, input.shape, max_input_samples)
                    )
                elif t is not None and input.shape[0] != t.size:
                    raise ValueError(
                        "Invalid shape for input sequence'{0}':{1}, row dimension must be equal to size of t, "
                        "i.e. '({2},*)')".format(input_name, input.shape, t.size)
                    )
            else:
                raise ValueError(
                    "Invalid shape for input sequence '{0}':{1}, column dimension must be equal to mld model dim "
                    "'{2}', i.e. '(*,{3})')".format(input_name, input.shape, req_input_dim_name, req_input_dim)
                )

        u = inputs_struct.u
        delta = inputs_struct.delta
        z = inputs_struct.z
        omega = inputs_struct.omega

        # Pre-build output arrays
        x_out = np.zeros((out_samples + 1, self.mld_info.n_states))
        y_out = np.zeros((out_samples, self.mld_info.n_outputs))
        con_out = np.zeros((out_samples, self.mld_info.n_cons), dtype=np.bool)
        t_out = _as2darray(np.linspace(0.0, stoptime, num=out_samples))

        # Check initial condition
        if x0 is None:
            x_out[0, :] = np.zeros((self.mld_info.n_states,))
        else:
            x_out[0, :] = np.asarray(x0).reshape(1, self.mld_info.n_states)

        # Pre-interpolate inputs into the desired time steps
        # todo interpolate inputs

        # return x_out, u_dt, delta, z, omega
        # Simulate the system
        for k in range(0, out_samples):
            x_out[k + 1, :] = (
                    self.A @ x_out[k, :] + self.B1 @ u[k, :] + self.B2 @ delta[k, :]
                    + self.B3 @ z[k, :] + self.B4 @ omega[k, :] + self.b5.T)
            y_out[k, :] = (
                    self.C @ x_out[k, :] + self.D1 @ u[k, :] + self.D2 @ delta[k, :]
                    + self.D3 @ z[k, :] + self.D4 @ omega[k, :] + self.d5.T)
            con_out[k, :] = (
                    (self.E1 @ x_out[k, :] + self.E2 @ u[k, :] + self.E3 @ delta[k, :]
                     + self.E4 @ z[k, :] + self.E5 @ omega[k, :]) <= self.g6.T)

        x_out = x_out[:-1, :]  # remove last point for equal length output arrays.

        return OrderedStructDict(t_out=t_out, y_out=y_out, x_out=x_out, con_out=con_out)

    def to_numeric(self, param_struct=None):
        if param_struct is None:
            param_struct = {}

        if self.mld_type == MldType.callable:
            mld_numeric_dict = {}
            for key, mat_eval in self.items():
                mld_numeric_dict[key] = mat_eval(param_struct)
        elif self.mld_type == MldType.symbolic:
            print("Performance warning, mld_type is not callable had to convert to callable")
            eval_mld = self.to_eval()
            return eval_mld.to_numeric(param_struct=param_struct)
        else:
            mld_numeric_dict = _deepcopy(dict(self))

        var_types_struct = _deepcopy(self.mld_info.var_types_struct)

        return MldModel(mld_numeric_dict, var_types_struct=var_types_struct)

    def to_eval(self):
        if self.mld_type in (MldType.numeric, MldType.symbolic):
            mld_eval_dict = {}
            for mat_id, expr in self.items():
                if not isinstance(expr, sp.Matrix):
                    expr = sp.Matrix(np.atleast_2d(expr))
                syms_tup, syms_str_tup = _get_syms_tup(expr)
                lam = sp.lambdify(syms_tup, expr, "numpy", dummify=False)
                mld_eval_dict[mat_id] = _lambda_wrapper(lam, syms_str_tup, wrapped_name="".join([mat_id, "_eval"]))

            var_types_struct = _deepcopy(self.mld_info.var_types_struct)
            eval_mld = MldModel(mld_eval_dict, var_types_struct=var_types_struct)
            return eval_mld
        else:
            return self

    def _set_mld_type(self, mld_data):
        is_callable = [callable(sys_mat) for sys_mat in mld_data.values()]
        if any(is_callable):
            if not all(is_callable):
                raise ValueError("If any mld matrices callable, all must be callable")
            else:
                self.mld_type = MldType.callable
        else:
            is_symbolic = [isinstance(sys_mat, (sp.Expr, sp.Matrix)) for sys_mat in mld_data.values()]
            if any(is_symbolic):
                self.mld_type = MldType.symbolic
            else:
                self.mld_type = MldType.numeric

    def _validate_sys_matrix_shapes(self, mld_data=None):
        if mld_data is None:
            mld_data = StructDict(_deepcopy(dict(self)))

        shapes_struct = _get_expr_shapes(mld_data)
        A_shape = shapes_struct.A
        C_shape = shapes_struct.C

        max_shapes = _MldMatType(
            *[_get_expr_shapes(*[mld_data.get(mat_id) for mat_id in mat_ids], get_max_dim=True)
              for mat_ids in self._sys_matrix_names]
        )

        if (A_shape[0] != A_shape[1]) and (0 not in A_shape):
            raise ValueError("Invalid shape for state matrix A:'{}', must be a square matrix or scalar".format(A_shape))

        sys_mat_zip = zip(self._sys_matrix_names.state_input, self._sys_matrix_names.output,
                          self._sys_matrix_names.constraint)

        val_err_fmt1 = ("Invalid shape for {mat_type} matrix/vector '{mat_id}':{mat_shape}, must have same {dim_type} "
                        "dimension as {req_mat_type} matrix '{req_mat_id}', i.e. '{req_shape}'").format
        val_err_fmt2 = ("Invalid {dim_type} dimension for {mat_type} matrix/vector '{mat_id}':{mat_shape}, must be "
                        "equal to maximum {dim_type} dimension of all {mat_type} matrices/vectors, i.e. "
                        "'{req_shape}'").format
        row_format = ("({0}, *)").format
        col_format = ("(*, {0})").format

        for sys_mat_id_triad in sys_mat_zip:
            state_input_matrix_id, out_matrix_id, con_matrix_id = sys_mat_id_triad
            state_input_matrix_shape = shapes_struct[state_input_matrix_id]
            out_matrix_shape = shapes_struct[out_matrix_id]
            con_matrix_shape = shapes_struct[con_matrix_id]
            if 0 not in state_input_matrix_shape:
                if state_input_matrix_shape[0] != A_shape[0] and (0 not in A_shape):
                    raise ValueError(
                        val_err_fmt1(mat_type='state/input', mat_id=state_input_matrix_id,
                                     mat_shape=state_input_matrix_shape, dim_type='row', req_mat_type='state',
                                     req_mat_id='A', req_shape=row_format(A_shape[0])))
                elif state_input_matrix_shape[0] != max_shapes.state_input[0] and state_input_matrix_id != 'A':
                    raise ValueError(
                        val_err_fmt2(dim_type='row', mat_type='state/input', mat_id=state_input_matrix_id,
                                     mat_shape=state_input_matrix_shape,
                                     req_shape=row_format(max_shapes.state_input[0])))
            if 0 not in out_matrix_shape:
                if out_matrix_shape[1] != state_input_matrix_shape[1] and (0 not in state_input_matrix_shape):
                    raise ValueError(
                        val_err_fmt1(mat_type='output', mat_id=out_matrix_id, mat_shape=out_matrix_shape,
                                     dim_type='column', req_mat_type='state/input', req_mat_id=state_input_matrix_id,
                                     req_shape=col_format(state_input_matrix_shape[1])))
                elif out_matrix_shape[0] != C_shape[0] and (0 not in C_shape):
                    raise ValueError(
                        val_err_fmt1(mat_type='output', mat_id=out_matrix_id, mat_shape=out_matrix_shape,
                                     dim_type='row', req_mat_type='output', req_mat_id='C',
                                     req_shape=row_format(C_shape[0])))
                elif out_matrix_shape[0] != max_shapes.output[0] and out_matrix_id != 'C':
                    raise ValueError(
                        val_err_fmt2(dim_type='row', mat_type='output', mat_id=out_matrix_id,
                                     mat_shape=out_matrix_shape, req_shape=row_format(max_shapes.output[0])))
            if 0 not in con_matrix_shape:
                if con_matrix_shape[1] != state_input_matrix_shape[1] and (0 not in state_input_matrix_shape):
                    raise ValueError(
                        val_err_fmt1(mat_type='constraint', mat_id=con_matrix_id, mat_shape=con_matrix_shape,
                                     dim_type='column', req_mat_type='state/input', req_mat_id=state_input_matrix_id,
                                     req_shape=col_format(state_input_matrix_shape[1])))
                elif con_matrix_shape[1] != out_matrix_shape[1] and (0 not in out_matrix_shape):
                    raise ValueError(
                        val_err_fmt1(mat_type='constraint', mat_id=con_matrix_id, mat_shape=con_matrix_shape,
                                     dim_type='column', req_mat_type='output', req_mat_id=out_matrix_id,
                                     req_shape=col_format(out_matrix_shape[1])))
                elif con_matrix_shape[0] != max_shapes.constraint[0]:
                    raise ValueError(
                        val_err_fmt2(dim_type='row', mat_type='constraint', mat_id=con_matrix_id,
                                     mat_shape=con_matrix_shape, req_shape=row_format(max_shapes.constraint[0])))

            max_shape_triad = np.maximum.reduce([state_input_matrix_shape, out_matrix_shape, con_matrix_shape])
            for max_shape, mat_id in zip(max_shapes, sys_mat_id_triad):
                if 0 in shapes_struct[mat_id] and not callable(mld_data[mat_id]):
                    if max_shape_triad[1]:
                        mld_data[mat_id] = np.zeros((max_shape[0], max_shape_triad[1]))
                    else:
                        mld_data[mat_id] = mld_data[mat_id].reshape(max_shape[0], 0)

        for vect_id in self._offset_vect_names:
            shape_vect = shapes_struct[vect_id]
            if not np.isin(shape_vect, (0, 1)).any():
                raise ValueError(
                    "'{0}' must be of type vector, scalar or null array, currently has shape:{1}".format(
                        vect_id, shape_vect)
                )

        if max_shapes.constraint and (0 in shapes_struct[self._offset_vect_names.constraint]):
            raise ValueError(
                "Constraint vector '{}' can only be null if all constraint matrices are null.".format(
                    self._offset_vect_names.constraint)
            )

        return mld_data

    def _get_all_syms_str_list(self):
        sym_str_set = {str(sym) for expr in self.values() if isinstance(expr, (sp.Expr, sp.Matrix)) for sym in
                       expr.free_symbols}
        return sorted(sym_str_set)

    @staticmethod
    def concat_mld(mld_model_list, sparse=True):
        concat_sys_mats = StructDict.fromkeys(MldModel._allowed_data_set, [])
        for sys_matrix_id in concat_sys_mats:
            concat_mat_list = []
            for model in mld_model_list:
                concat_mat_list.append(model[sys_matrix_id])
            if sys_matrix_id[0].isupper():
                if sparse:
                    concat_sys_mats[sys_matrix_id] = scs.block_diag(concat_mat_list)
                else:
                    concat_sys_mats[sys_matrix_id] = scl.block_diag(*concat_mat_list)
            else:
                concat_sys_mats[sys_matrix_id] = np.vstack(concat_mat_list)

        concat_var_type_info = StructDict.fromkeys(MldInfo._var_type_names)
        for var_type_name in concat_var_type_info:
            concat_info_list = []
            for model in mld_model_list:
                concat_info_list.append(model.mld_info[var_type_name])

            concat_var_type_info[var_type_name] = np.hstack(concat_info_list)

        concat_var_type_info.var_type_delta = None
        concat_var_type_info.var_type_z = None

        concat_mld = MldModel(concat_sys_mats, var_types_struct=concat_var_type_info)

        return concat_mld


class MldInfo(MldBase):
    __internal_names = ['_mld_model']
    _internal_names_set = MldBase._internal_names_set.union(__internal_names)

    _valid_var_types = ['c', 'b']

    _metadata_names = ['dt']

    _state_names_user_setable = ['x']
    _input_names_user_setable = ['u', 'omega']
    _input_names_non_setable = ['delta', 'z']

    _state_names = _state_names_user_setable
    _input_names = _input_names_user_setable + _input_names_non_setable
    _state_input_names = _state_names + _input_names
    _state_input_names_non_setable = _input_names_non_setable

    _state_input_dim_names = ["".join(['n', name]) for name in _state_input_names]
    _sys_dim_names = ['n_states', 'n_outputs', 'n_cons']
    _var_type_names = ["".join(['var_type_', name]) for name in _state_input_names]
    _bin_dim_names = ["".join([dim_name, '_l']) for dim_name in _state_input_dim_names]
    _bin_dim_names_non_setable = ["".join([dim_name, '_l']) for dim_name in _state_input_names_non_setable]

    _allowed_data_set = set(
        _metadata_names + _state_input_dim_names + _sys_dim_names + _var_type_names + _bin_dim_names)

    @_process_args_decor()
    def __init__(self, *args, mld_model=None, dt=None, bin_dims_struct=None, var_types_struct=None, **kwargs):
        super(MldInfo, self).__init__(*args, **kwargs)
        self._mld_model = None
        self.update(*args, mld_model=mld_model, dt=dt, bin_dims_struct=bin_dims_struct,
                    var_types_struct=var_types_struct, **kwargs)

    def __reduce__(self):
        mld_data = list(self.mld_model.items()) if isinstance(self.mld_model, MldModel) else None
        if hasattr(self, '_key'):
            args = (self._key, mld_data, None, self.var_types_struct)
        else:
            args = (mld_data, None, self.var_types_struct)

        return (self.__class__, args)

    @property
    def var_types_struct(self):
        return StructDict(
            {var_type_name: self.get(var_type_name) for var_type_name in self._var_type_names})

    @property
    def mld_model(self):
        return self._mld_model

    @mld_model.setter
    def mld_model(self, value):
        raise AttributeError("Cannot set mld_model directly use MldInfo.update() instead.")

    @_process_args_decor(update_info_structs=True)
    def update(self, *args, mld_model=None, dt=None, bin_dims_struct=None, var_types_struct=None, **kwargs):
        if args and isinstance(args[0], self.__class__):
            self._sdict_update(args[0])
            self._mld_model = args[0].mld_model
            mld_model = None
        elif not isinstance(mld_model, MldModel):
            try:
                mld_model = MldModel(dict(mld_model))
                mld_model.mld_info = self
            except Exception as e:
                mld_model = None

        self._mld_model = mld_model or self._mld_model

        if self.mld_model:
            self._set_var_dims()
            self._set_metadata(dt=dt)
            self._set_var_info(bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct)

    def _set_metadata(self, dt=None):
        dt_queue = [dt, self.dt]
        dt = next((item for item in dt_queue if item is not None), 0)
        self._sdict_update(dt=dt)

    def _set_var_info(self, bin_dims_struct=None, var_types_struct=None):
        # either set to new value, old value, or zero in that order - never None
        bin_dims_struct = bin_dims_struct or {}
        var_types_struct = var_types_struct or {}

        new_var_info = StructDict(self.items())
        zip_gen = zip(self._state_input_dim_names, self._bin_dim_names, self._var_type_names)

        for (state_input_dim_name, bin_dim_name, var_type_name) in zip_gen:
            bin_dim = bin_dims_struct.get(bin_dim_name)
            var_type = var_types_struct.get(var_type_name)
            state_input_dim = self.get(state_input_dim_name)

            if bin_dim_name in self._bin_dim_names_non_setable:
                if bin_dim:
                    raise ValueError(
                        "Cannot manually set ndelta_l, nz_l - these are fixed by the MLD specification and dimension.")
                elif state_input_dim_name == 'ndelta':
                    bin_dim = self[state_input_dim_name]
                    new_var_info[bin_dim_name] = bin_dim
                elif state_input_dim_name == 'nz':
                    bin_dim = 0
                    new_var_info[bin_dim_name] = bin_dim
            else:
                bin_dim_queue = [self._get_num_var_bin(var_type), bin_dim, self.get(bin_dim_name)]
                bin_dim = next((item for item in bin_dim_queue if item is not None), 0)
                new_var_info[bin_dim_name] = bin_dim

            if var_type is not None:
                var_type = self._check_var_types_vect_valid(var_type, new_var_info, bin_dim_name)
                if var_type.size != self[state_input_dim_name]:
                    raise ValueError(
                        "Dimension of '{0}' must match dimension: '{1}'".format(var_type_name, state_input_dim_name))
                new_var_info[var_type_name] = var_type
            else:
                try:
                    new_var_info[var_type_name] = np.hstack(
                        [np.repeat('c', state_input_dim - bin_dim), np.repeat('b', bin_dim)])
                except ValueError:
                    raise ValueError(
                        "Value of '{0}':{1} must be non-negative value <= dimension '{2}':{3}".format(
                            bin_dim_name, bin_dim, state_input_dim_name, self[state_input_dim_name]))

        self._sdict_update(new_var_info)

    def _check_var_types_vect_valid(self, var_types_vect, mld_info_data=None, bin_dim_name=None):
        if var_types_vect is None:
            return var_types_vect
        else:
            var_types_vect = np.ravel(var_types_vect)
            if not np.setdiff1d(var_types_vect, self._valid_var_types).size == 0:
                raise ValueError('All elements of var_type_vectors must be in {}'.format(self._valid_var_types))
            if mld_info_data and bin_dim_name and (
                    mld_info_data.get(bin_dim_name) != self._get_num_var_bin(var_types_vect)):
                raise ValueError(
                    "Number of binary variables in var_type_vect:'{0}', does not match dimension of '{1}':{2}".format(
                        var_types_vect, bin_dim_name, mld_info_data.get(bin_dim_name)))

            return var_types_vect

    def _set_var_dims(self):
        for dim_name, sys_matrix_ids in _mld_dim_map.items():
            system_matrices = (self.mld_model[sys_id] for sys_id in sys_matrix_ids)
            if dim_name not in self._sys_dim_names:
                self._sdict_setitem(dim_name, _get_expr_shapes(*system_matrices, get_max_dim=True)[1])
            else:
                self._sdict_setitem(dim_name, _get_expr_shapes(*system_matrices, get_max_dim=True)[0])

    @classmethod
    def _update_info_structs_from_kwargs(cls, kwargs=None):
        bin_dims_struct = kwargs.get('bin_dims_struct') or StructDict()
        var_types_struct = kwargs.get('var_types_struct') or StructDict()

        kwargs = kwargs or {}
        for key in cls._bin_dim_names:
            if key in kwargs:
                bin_dims_struct[key] = kwargs.pop(key)
        for key in cls._var_type_names:
            if key in kwargs:
                var_types_struct[key] = kwargs.pop(key)

        kwargs['bin_dims_struct'] = bin_dims_struct if bin_dims_struct else None
        kwargs['var_types_struct'] = var_types_struct if var_types_struct else None

        return kwargs

    @staticmethod
    def _get_num_var_bin(var_types_vect):
        if var_types_vect is None:
            return None
        else:
            var_types_vect_flat = np.ravel(var_types_vect)
            return (var_types_vect_flat == 'b').sum()


def _as2darray(array):
    if array is None:
        out_array = np.empty(shape=(0, 0))
    else:
        out_array = np.array(array)
        if len(out_array.shape) == 1:  # return column array if 1 dim
            out_array = out_array[:, np.newaxis]
    return out_array


def _get_expr_shape(expr):
    if expr is None:
        return (0, 0)
    elif np.isscalar(expr) or isinstance(expr, sp.Expr):
        return (1, 1)
    else:
        try:
            expr_shape = expr.shape
            if 0 in expr_shape:
                return (0, 0)
            if len(expr_shape) == 1:
                return (expr_shape[0], 1)
            elif len(expr_shape) == 2:
                return expr_shape
            else:
                raise NotImplementedError("Maximum supported dimension is 2, got {}".format(len(expr_shape)))
        except AttributeError:
            pass

    if callable(expr):
        try:
            return _get_expr_shape(expr(empty_call=True))
        except Exception:
            try:
                args = [1] * len(inspect.getfullargspec(expr).args)
                return _get_expr_shape(expr(*args))
            except Exception as e:
                raise e
    else:
        raise TypeError("Invalid expression type: '{0}', for expr: '{1!s}'".format(type(expr), expr))


def _get_expr_shapes(*args, get_max_dim=False):
    if not args:
        return None

    shapes = []
    if isinstance(args[0], dict):
        shape_struct = StructDict()
        for expr_id, expr in args[0].items():
            expr_shape = _get_expr_shape(expr)
            shapes.append(expr_shape)
            shape_struct[expr_id] = _get_expr_shape(expr)
        if not get_max_dim:
            return shape_struct
    else:
        for arg in args:
            shapes.append(_get_expr_shape(arg))
    if get_max_dim:
        return tuple(np.maximum.reduce(shapes))
    else:
        return shapes


def _get_syms_tup(expr):
    try:
        sym_dict = {str(sym): sym for sym in expr.free_symbols}
        sym_str_list = sorted(sym_dict.keys())
        sym_list = [sym_dict.get(sym) for sym in sym_str_list]
    except AttributeError:
        return tuple(), tuple()

    return tuple(sym_list), tuple(sym_str_list)


# todo functions should be able to take dictionary or positional arguments
def _lambda_wrapper(func, local_syms_str, wrapped_name=None):
    if wrapped_name:
        func.__qualname__ = wrapped_name
        func.__name__ = wrapped_name

    @functools.wraps(func)
    def wrapped(*args, param_struct=None, **kwargs):
        arg_list = []
        if kwargs.pop('empty_call', False):
            arg_list = [1] * len(local_syms_str)
        elif args and isinstance(args[0], dict):
            param_struct = args[0]
        elif len(args) == len(local_syms_str) and param_struct is None:
            arg_list = args

        if isinstance(param_struct, dict):
            try:
                arg_list = [param_struct[sym_str] for sym_str in local_syms_str]
            except TypeError:
                raise ValueError("param_struct must be dictionary like.")
            except KeyError:
                raise KeyError("Incorrect param_struct supplied, requires dict with keys {}".format(local_syms_str))

        return func(*arg_list)

    return wrapped


if __name__ == '__main__':
    a = dict.fromkeys(MldModel._allowed_data_set, np.ones((2, 2)))
    a.update(b5=np.ones((2, 1)), d5=np.ones((2, 1)), g6=np.ones((2, 1)))
    mld = MldModel(a)
    mld2 = MldModel({'A': 1})
    #
    # from models.model_generators import DewhModelGenerator
    # b = DewhModelGenerator().gen_dewh_symbolic_mld_sys_matrices()
    # c = b.to_eval()
