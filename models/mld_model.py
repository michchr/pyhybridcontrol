import functools
import inspect
import wrapt
from collections import OrderedDict, namedtuple as NamedTuple
from copy import deepcopy as _deepcopy
import numpy as np
import scipy.linalg as scl
import scipy.sparse as scs
import sympy as sp

from utils.structdict import StructDict, OrderedStructDict, struct_repr


def _process_args_decor(update_info_structs=False):
    @wrapt.decorator
    def wrapper(func, self, args, kwargs):
        if update_info_structs:
            kwargs = MldInfo._update_info_structs_from_kwargs(kwargs=kwargs)
        return func(*args, **kwargs)

    return wrapper


class MldBase(StructDict):
    __internal_names = []
    _internal_names_set = StructDict._internal_names_set.union(__internal_names)

    _data_types = ()
    _data_layout = {}
    _allowed_data_set = set()

    def __init__(self, *args, **kwargs):
        super(MldBase, self).__init__(dict.fromkeys(self._allowed_data_set))

    def __setitem__(self, key, value):
        if key in self._allowed_data_set:
            self.update(**{key: value})
        elif self._internal_names_set and key in self._internal_names_set:
            object.__setattr__(self, key, value)
        else:
            raise KeyError("key: '{}' is not in _data_headings_set or _allowed_data_set.".format(key))

    def _sdict_setitem(self, key, value):
        super(MldBase, self).__setitem__(key, value)

    def clear(self):
        super(MldBase, self).update(dict.fromkeys(self._allowed_data_set))

    def get_sub_struct(self, keys):
        return StructDict(self.get_sub_dict(keys))

    def __delitem__(self, key):
        super(MldBase, self).__setitem__(key, None)

    def __repr__(self):
        repr_dict = {data_type: self.get_sub_struct(self._data_layout[data_type]) for data_type in
                     self._data_types}
        return struct_repr(repr_dict, type_name=self.__class__.__name__)


_MLD_MODEL_TYPE_NAMES = ['numeric_mld', 'symbolic_mld', 'callable_mld']
_MldModelTypesNamedTup = NamedTuple('MldTypesNamedTup', _MLD_MODEL_TYPE_NAMES)
MldModelTypes = _MldModelTypesNamedTup._make(_MLD_MODEL_TYPE_NAMES)

_MLD_MODEL_MAT_TYPES_NAMES = ['state_input_mats', 'output_mats', 'constraint_mats']
_MldModelMatTypesNamedTup = NamedTuple('MldMatTypesNamedTup', ['state_input', 'output', 'constraint'])


class MldModel(MldBase):
    __internal_names = ['_mld_info', '_mld_type', '_shapes_struct']
    _internal_names_set = MldBase._internal_names_set.union(__internal_names)

    MldModelMatTypes = _MldModelMatTypesNamedTup._make(_MLD_MODEL_MAT_TYPES_NAMES)

    _data_types = MldModelMatTypes
    _data_layout = OrderedDict([(MldModelMatTypes.state_input, ['A', 'B1', 'B2', 'B3', 'B4', 'b5']),
                                (MldModelMatTypes.output, ['C', 'D1', 'D2', 'D3', 'D4', 'd5']),
                                (MldModelMatTypes.constraint, ['E1', 'E2', 'E3', 'E4', 'E5', 'g6'])])

    _allowed_data_set = set([data for data_type in _data_layout.values() for data in data_type])

    _sys_matrix_names = _data_layout
    _offset_vect_names = _MldModelMatTypesNamedTup(*[mat_names[-1] for mat_names in _sys_matrix_names.values()])

    def __init__(self, system_matrices=None, dt=None, bin_dims_struct=None, var_types_struct=None, *args, **kwargs):
        super(MldModel, self).__init__(*args, **kwargs)
        super(MldModel, self).update(dict.fromkeys(self._allowed_data_set, np.empty(shape=(0, 0))))
        self._mld_info = MldInfo()
        self._mld_type = None
        self._shapes_struct = None
        self.update(*args, system_matrices=system_matrices, dt=dt, bin_dims_struct=bin_dims_struct,
                    var_types_struct=var_types_struct, from_init=True, **kwargs)

    @property
    def mld_info(self):
        return self._mld_info

    @mld_info.setter
    def mld_info(self, mld_info):
        if isinstance(mld_info, MldInfo):
            self._mld_info = mld_info
        else:
            raise TypeError("mld_info must be of type '{}'".format(self.mld_info.__class__.__name__))

    @property
    def shapes_struct(self):
        return self._shapes_struct

    @property
    def mld_type(self):
        return self._mld_type

    @_process_args_decor(update_info_structs=True)
    def update(self, *args, system_matrices=None, dt=None, bin_dims_struct=None, var_types_struct=None, **kwargs):
        from_init = kwargs.pop('from_init', False)
        if system_matrices and kwargs:
            raise ValueError("Individual matrix arguments cannot be set if 'system_matrices' argument is set")

        if args and isinstance(args[0], self.__class__):
            self.mld_info = args[0].mld_info

        mld_info_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if
                           key in self.mld_info._allowed_data_set}

        creation_matrices = system_matrices or kwargs
        if not isinstance(creation_matrices, dict):
            try:
                creation_matrices = dict(creation_matrices)
            except Exception as exc:
                raise exc

        new_sys_mats = dict(self.items())
        try:
            for sys_matrix_id, system_matrix in creation_matrices.items():
                if sys_matrix_id in self._allowed_data_set:
                    old_val = self.get(sys_matrix_id)
                    if isinstance(system_matrix, (sp.Expr)):
                        system_matrix = sp.Matrix([system_matrix])
                    elif not isinstance(system_matrix, sp.Matrix) and not callable(system_matrix):
                        system_matrix = np.atleast_2d(system_matrix)
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
                new_sys_mats['C'] = sp.Matrix(np.eye(*(new_sys_mats['A'].shape)))
            else:
                new_sys_mats['C'] = np.eye(*(new_sys_mats['A'].shape))

        try:
            new_sys_mats, shapes_struct = self._validate_sys_matrix_shapes(new_sys_mats)
            self._set_mld_type(new_sys_mats)
        except ValueError as ve:
            raise ve

        super(MldModel, self).update(new_sys_mats)
        self._shapes_struct = shapes_struct

        if self.mld_type == MldModelTypes.symbolic_mld:
            required_params = self._get_all_syms_str_list()
        else:
            required_params = None

        self.mld_info.update(self, dt=dt, bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct,
                             required_params=required_params, **mld_info_kwargs)

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

        if self.mld_type == MldModelTypes.callable_mld:
            mld_numeric_dict = {}
            for key, mat_eval in self.items():
                mld_numeric_dict[key] = mat_eval(param_struct)
        elif self.mld_type == MldModelTypes.symbolic_mld:
            print("Performance warning, mld_type is not callable had to convert to callable")
            eval_mld = self.to_eval()
            return eval_mld.to_numeric(param_struct=param_struct)
        else:
            mld_numeric_dict = _deepcopy(dict(self))

        var_types_struct = _deepcopy(self.mld_info.var_types_struct)

        return MldModel(mld_numeric_dict, var_types_struct=var_types_struct)

    def to_eval(self):
        if self.mld_type in (MldModelTypes.numeric_mld, MldModelTypes.symbolic_mld):
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

    def __reduce__(self):
        s_reduce = super(MldModel, self).__reduce__()
        reduce = (s_reduce[0], s_reduce[1] + (self.mld_info.dt, None, self.mld_info.var_types_struct.as_dict()))
        return reduce

    def _set_mld_type(self, mld_data):
        is_callable = [callable(sys_mat) for sys_mat in mld_data.values()]
        if any(is_callable):
            if not all(is_callable):
                raise ValueError("If any mld matrices callable, all must be callable")
            else:
                self._mld_type = MldModelTypes.callable_mld
        else:
            is_symbolic = [isinstance(sys_mat, (sp.Expr, sp.Matrix)) for sys_mat in mld_data.values()]
            if any(is_symbolic):
                self._mld_type = MldModelTypes.symbolic_mld
            else:
                self._mld_type = MldModelTypes.numeric_mld

    def _validate_sys_matrix_shapes(self, mld_data=None):
        if mld_data is None:
            mld_data = StructDict(_deepcopy(dict(self)))

        shapes_struct = _get_expr_shapes(mld_data)
        A_shape = shapes_struct['A']
        C_shape = shapes_struct['C']

        max_shape_mat_types = _MldModelMatTypesNamedTup._make(
            [_get_max_shape(shapes_struct.get_sub_dict(self._sys_matrix_names[mat_type]).values()) for mat_type
             in self.MldModelMatTypes])

        if (A_shape[0] != A_shape[1]) and (0 not in A_shape):
            raise ValueError("Invalid shape for state matrix A:'{}', must be a square matrix or scalar".format(A_shape))

        sys_mat_ids_zip = zip(self._sys_matrix_names[self.MldModelMatTypes.state_input],
                              self._sys_matrix_names[self.MldModelMatTypes.output],
                              self._sys_matrix_names[self.MldModelMatTypes.constraint])

        val_err_fmt1 = ("Invalid shape for {mat_type} matrix/vector '{mat_id}':{mat_shape}, must have same {dim_type} "
                        "dimension as {req_mat_type} matrix '{req_mat_id}', i.e. '{req_shape}'").format
        val_err_fmt2 = ("Invalid shape for {mat_type} matrix/vector '{mat_id}':{mat_shape}, {dim_type} dimension must "
                        "be equal to maximum {dim_type} dimension of all {mat_type} matrices/vectors, i.e. "
                        "'{req_shape}'").format
        row_format = ("({0}, *)").format
        col_format = ("(*, {0})").format

        for sys_mat_id_triad in sys_mat_ids_zip:
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
                elif state_input_matrix_shape[0] != max_shape_mat_types.state_input[0] and state_input_matrix_id != 'A':
                    raise ValueError(
                        val_err_fmt2(dim_type='row', mat_type='state/input', mat_id=state_input_matrix_id,
                                     mat_shape=state_input_matrix_shape,
                                     req_shape=row_format(max_shape_mat_types.state_input[0])))
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
                elif out_matrix_shape[0] != max_shape_mat_types.output[0] and out_matrix_id != 'C':
                    raise ValueError(
                        val_err_fmt2(dim_type='row', mat_type='output', mat_id=out_matrix_id,
                                     mat_shape=out_matrix_shape, req_shape=row_format(max_shape_mat_types.output[0])))
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
                elif con_matrix_shape[0] != max_shape_mat_types.constraint[0]:
                    raise ValueError(
                        val_err_fmt2(dim_type='row', mat_type='constraint', mat_id=con_matrix_id,
                                     mat_shape=con_matrix_shape,
                                     req_shape=row_format(max_shape_mat_types.constraint[0])))

            max_shape_triad = _get_max_shape([state_input_matrix_shape, out_matrix_shape, con_matrix_shape])
            for max_shape, mat_id in zip(max_shape_mat_types, sys_mat_id_triad):
                if 0 in shapes_struct[mat_id] and not callable(mld_data[mat_id]):
                    if max_shape_triad[1]:
                        mld_data[mat_id] = np.zeros((max_shape[0], max_shape_triad[1]))
                    else:
                        mld_data[mat_id] = mld_data[mat_id].reshape(max_shape[0], 0)

                    shapes_struct[mat_id] = _get_expr_shape(mld_data[mat_id])

        for vect_id in self._offset_vect_names:
            shape_vect = shapes_struct[vect_id]
            if not np.isin(shape_vect, (0, 1)).any():
                raise ValueError(
                    "'{0}' must be of type vector, scalar or null array, currently has shape:{1}".format(
                        vect_id, shape_vect)
                )

        if any(max_shape_mat_types.constraint) and (0 in shapes_struct[self._offset_vect_names.constraint]):
            raise ValueError(
                "Constraint vector '{}' can only be null if all constraint matrices are null.".format(
                    self._offset_vect_names.constraint)
            )

        return mld_data, shapes_struct

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


_MLD_INFO_TYPE_NAMES = ['sys_dims', 'state_input_dims', 'bin_dims', 'var_types', 'meta_data']
_MldInfoTypesNamedTup = NamedTuple('MldInfoTypesNamedTup',
                                   ['sys_dims', 'state_input_dims', 'bin_dims', 'var_types', 'meta_data'])


class MldInfo(MldBase):
    __internal_names = ['_mld_model']
    _internal_names_set = MldBase._internal_names_set.union(__internal_names)

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

    MldInfoTypes = _MldInfoTypesNamedTup._make(_MLD_INFO_TYPE_NAMES)

    _valid_var_types = ['c', 'b']

    _sys_dim_names = ['n_states', 'n_outputs', 'n_cons']
    _meta_data_names = ['dt', 'required_params']

    _state_names_user_setable = ['x']
    _input_names_user_setable = ['u', 'omega']
    _input_names_non_setable = ['delta', 'z']

    _state_names = ['x']
    _input_names = ['u', 'delta', 'z', 'omega']
    _state_input_names = _state_names + _input_names
    _state_input_names_non_setable = ['delta', 'z']

    _state_input_dim_names = ["".join(['n', name]) for name in _state_input_names]

    _bin_dim_names = ["".join([dim_name, '_l']) for dim_name in _state_input_dim_names]
    _bin_dim_names_non_setable = ["".join(['n', dim_name, '_l']) for dim_name in _state_input_names_non_setable]

    _var_type_names = ["".join(['var_type_', name]) for name in _state_input_names]

    _data_types = MldInfoTypes
    _data_layout = OrderedDict([(MldInfoTypes.sys_dims, _sys_dim_names),
                                (MldInfoTypes.state_input_dims, _state_input_dim_names),
                                (MldInfoTypes.bin_dims, _bin_dim_names),
                                (MldInfoTypes.var_types, _var_type_names),
                                (MldInfoTypes.meta_data, _meta_data_names)])
    _allowed_data_set = set([data for data_type in _data_layout.values() for data in data_type])

    def __init__(self, mld_data=None, dt=None, bin_dims_struct=None, var_types_struct=None, required_params=None,
                 **kwargs):
        super(MldInfo, self).__init__()
        self.update(mld_data=mld_data, dt=dt, bin_dims_struct=bin_dims_struct,
                    var_types_struct=var_types_struct, required_params=required_params, **kwargs)

    def __reduce__(self):
        mld_data = dict(self.mld_model) if isinstance(self.mld_model, MldModel) else None
        if hasattr(self, '_key'):
            args = (self._key, mld_data, None, self.var_types_struct)
        else:
            args = (mld_data, None, self.var_types_struct)
        return (self.__class__, args)

    @property
    def var_types_struct(self):
        return self.get_sub_struct(self._data_layout[self.MldInfoTypes.var_types])

    @_process_args_decor(update_info_structs=True)
    def update(self, mld_data=None, dt=None, bin_dims_struct=None, var_types_struct=None, required_params=None,
               **kwargs):

        non_updateable = self._allowed_data_set.intersection(kwargs)
        if non_updateable:
            raise ValueError(
                "Cannot set values for keys: '{}' directly, these are updated automatically based on the MldModel "
                "structure and other valid parameters".format(non_updateable))
        elif kwargs:
            raise ValueError(
                "Invalid kwargs supplied: '{}'".format(kwargs)
            )

        shapes_struct = None
        if mld_data is not None:
            if isinstance(mld_data, self.__class__):
                super(MldInfo, self).update(mld_data)
            elif isinstance(mld_data, MldModel):
                shapes_struct = mld_data.shapes_struct
            elif isinstance(mld_data, dict):
                missing_keys = set(mld.keys()).difference(MldModel._allowed_data_set)
                if missing_keys:
                    raise ValueError(
                        "If mld_data is a shape_struct require entry for each matrix/vector in an MldModel, "
                        "missing these keys: {}".format(missing_keys))
                elif not all([(isinstance(value, tuple) and len(value) == 2) for value in mld_data.values()]):
                    raise ValueError(
                        "If mld_data is a shape_struct for mld_model all values must be tuples of length 2.")
                shapes_struct = mld_data
            else:
                raise TypeError("Invalid type for mld_data.")

            self._set_metadata(dt=dt, required_params=required_params)
            self._set_var_dims(shapes_struct)
            self._set_var_info(bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct)

    def _set_metadata(self, dt=None, required_params=None):
        dt_queue = [dt, self.get('dt')]
        dt = next((item for item in dt_queue if item is not None), 0)
        super(MldInfo, self).update(dt=dt, required_params=required_params)

    def _set_var_info(self, bin_dims_struct=None, var_types_struct=None):
        # either set to new value, old value, or zero in that order - never None
        bin_dims_struct = bin_dims_struct or {}
        var_types_struct = var_types_struct or {}

        new_var_info = dict(self.items())
        zip_gen = zip(self._state_input_names, self._state_input_dim_names, self._bin_dim_names, self._var_type_names)

        for (state_input_name, state_input_dim_name, bin_dim_name, var_type_name) in zip_gen:
            bin_dim = bin_dims_struct.get(bin_dim_name)
            var_type = var_types_struct.get(var_type_name)
            state_input_dim = self.get(state_input_dim_name)

            if state_input_name in self._state_input_names_non_setable:
                if bin_dim:
                    raise ValueError(
                        "Cannot manually set ndelta_l, nz_l - these are fixed by the MLD specification and dimension.")
                elif state_input_name == 'delta':
                    new_var_info[bin_dim_name] = bin_dim = self[state_input_dim_name]
                elif state_input_name == 'z':
                    new_var_info[bin_dim_name] = bin_dim = 0
            else:
                bin_dim_queue = [self._get_num_var_bin(var_type), bin_dim, self.get(bin_dim_name)]
                bin_dim = next((item for item in bin_dim_queue if item is not None), 0)
                new_var_info[bin_dim_name] = bin_dim

            if var_type is not None:
                var_type = self._check_var_types_vect_valid(var_type, new_var_info, bin_dim_name)
                if var_type.size != self[state_input_dim_name]:
                    raise ValueError(
                        "Dimension of '{0}' must match dimension: '{1}'".format(var_type_name, state_input_dim_name))
                else:
                    new_var_info[var_type_name] = var_type
            else:
                try:
                    new_var_info[var_type_name] = np.hstack(
                        [np.repeat('c', state_input_dim - bin_dim), np.repeat('b', bin_dim)])
                except ValueError:
                    raise ValueError(
                        "Value of '{0}':{1} must be non-negative value <= dimension '{2}':{3}".format(
                            bin_dim_name, bin_dim, state_input_dim_name, self[state_input_dim_name]))

        super(MldInfo, self).update(new_var_info)

    def _check_var_types_vect_valid(self, var_types_vect, mld_info_data, bin_dim_name=None):
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

    def _set_var_dims(self, shapes_struct):
        if shapes_struct is not None:
            for dim_name, sys_matrix_ids in self._mld_dim_map.items():
                shapes = StructDict.get_sub_dict(shapes_struct, sys_matrix_ids)
                max_shape = _get_max_shape(shapes.values())
                if dim_name not in self._sys_dim_names:
                    self._sdict_setitem(dim_name, max_shape[1])
                else:
                    self._sdict_setitem(dim_name, max_shape[0])
        else:
            for dim_name in self._mld_dim_map:
                self._sdict_setitem(dim_name, self[dim_name] or 0)

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
            # if 0 in expr_shape:
            #     return (0, 0)
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


def _get_max_shape(shapes):
    shapes = np.asarray(list(shapes))
    return tuple(np.maximum.reduce(shapes))


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
        return _get_max_shape(shapes)
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
def _lambda_wrapper(func, local_syms_str_tup, wrapped_name=None):
    if wrapped_name:
        func.__qualname__ = wrapped_name
        func.__name__ = wrapped_name

    @functools.wraps(func)
    def wrapped(*args, param_struct=None, **kwargs):
        arg_list = args
        if kwargs.pop('empty_call', False):
            arg_list = [np.NaN] * len(local_syms_str_tup)
        elif args and isinstance(args[0], dict) or isinstance(param_struct, dict):
            param_struct = args[0] or param_struct
            try:
                arg_list = [param_struct[sym_str] for sym_str in local_syms_str_tup]
            except KeyError:
                missing_keys = set(local_syms_str_tup).difference(param_struct.keys())
                raise ValueError(
                    "Required keys in param_struct are: '{}', the following keys are missing: '{}'".format(
                        local_syms_str_tup, missing_keys))

        return func(*arg_list)

    wrapped.__signature__ = inspect.signature(func)
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
