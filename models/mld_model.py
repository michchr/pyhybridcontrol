import functools
import inspect
from collections import OrderedDict, namedtuple as NamedTuple
from copy import deepcopy as _deepcopy, copy as _copy
from itertools import zip_longest

from operator import itemgetter as _itemgetter, methodcaller as _methodcaller
from builtins import property as _property

import numpy as np
import scipy.linalg as scl
import scipy.sparse as scs
import sympy as sp
import wrapt

from utils.structdict import StructDict, OrderedStructDict, struct_repr, StructDictMeta


class _MldMeta(StructDictMeta):
    def __new__(cls, name, bases, _dict):
        kls = super(_MldMeta, cls).__new__(cls, name, bases, _dict)

        def _itemsetter(name):
            def caller(self, value):
                self.__setitem__(name, value)
            return caller

        for name in getattr(kls, '_allowed_data_set'):
            setattr(kls, name, _property(fget=_itemgetter(name),
                                         fset=_itemsetter(name),
                                         doc=f"Alias for self['{name}']"))
        return kls

class MldBase(StructDict, metaclass=_MldMeta):
    __internal_names = []
    _internal_names_set = StructDict._internal_names_set.union(__internal_names)

    _data_types = ()
    _data_layout = {}
    _allowed_data_set = set()

    def __init__(self, *args, **kwargs):
        super(MldBase, self).__init__()
        super(MldBase, self).update(dict.fromkeys(self._allowed_data_set))

    def __setitem__(self, key, value):
        if key in self._allowed_data_set:
            self.update(**{key: value})
        else:
            raise KeyError("key: '{}' is not in _allowed_data_set.".format(key))

    def _sdict_setitem(self, key, value):
        super(MldBase, self).__setitem__(key, value)

    def clear(self):
        self.__init__()

    def pop(self, *args, **kwargs):
        raise PermissionError("Items can not be removed from mld_model.")

    def popitem(self, *args, **kwargs):
        raise PermissionError("Items can not be removed from mld_model.")

    def get_sub_struct(self, keys):
        return StructDict.sub_struct_fromdict(self, keys)

    def __delitem__(self, key):
        raise PermissionError("Items can not be removed from mld_model.")

    def __repr__(self):
        value_repr = lambda value: struct_repr(value, type_name='', repr_format_str='{type_name}{{{key_arg}{items}}}',
                                               align_values=True)
        repr_dict = {data_type: value_repr(self.get_sub_struct(self._data_layout[data_type])) for data_type in
                     self._data_types}
        return struct_repr(repr_dict, type_name=self.__class__.__name__, align_values=True, align_padding_width=1,
                           value_format_str='\b{value}')

    @classmethod
    def _constructor(cls, items, state, attribute_override=None, copy_items=False, copy_state=False, memo=None):
        obj = cls.__new__(cls)
        if copy_items:
            items = _deepcopy(items, memo=memo)
        if copy_state:
            state = _deepcopy(state, memo=memo)

        super(MldBase, obj).update(items)
        obj.__dict__.update(state)
        if attribute_override:
            obj.__dict__.update(attribute_override)

        return obj

    def copy(self):
        return self.__class__._constructor(self, self.__dict__)

    __copy__ = copy

    def deepcopy(self, memo=None):
        return self.__class__._constructor(dict(self), self.__dict__, copy_items=True, copy_state=True, memo=memo)

    __deepcopy__ = deepcopy

    def __reduce__(self):
        return (self.__class__._constructor, (dict(self), self.__dict__))


@wrapt.decorator
def _process_mld_args_decor(func, self, args, kwargs):
    bin_dims_struct = kwargs.get('bin_dims_struct') or {}
    var_types_struct = kwargs.get('var_types_struct') or {}
    if kwargs:
        bin_dim_args = set(MldInfo._var_to_bin_dim_names_map.values()).intersection(kwargs)
        var_types_args = set(MldInfo._var_to_type_names_map.values()).intersection(kwargs)
        bin_dims_struct.update({key: kwargs.pop(key) for key in bin_dim_args})
        var_types_struct.update({key: kwargs.pop(key) for key in var_types_args})

    kwargs['bin_dims_struct'] = bin_dims_struct if bin_dims_struct else None
    kwargs['var_types_struct'] = var_types_struct if var_types_struct else None

    return func(*args, **kwargs)


class MldInfo(MldBase):
    __internal_names = []
    _internal_names_set = MldBase._internal_names_set.union(__internal_names)

    _MLD_INFO_DATA_TYPE_NAMES = ['sys_dims', 'state_input_dims', 'output_dim', 'slack_dim', 'bin_dims', 'var_types',
                                 'meta_data']
    _MldInfoDataTypesNamedTup = NamedTuple('MldInfoTypesNamedTup', _MLD_INFO_DATA_TYPE_NAMES)
    MldInfoDataTypes = _MldInfoDataTypesNamedTup._make(_MLD_INFO_DATA_TYPE_NAMES)

    # MldInfo Config
    _valid_var_types = ['c', 'b']
    _meta_data_names = ['dt', 'param_struct', 'required_params']

    _sys_dim_names = ['n_states', 'n_outputs', 'n_cons']

    _state_names = ['x']
    _input_names = ['u', 'delta', 'z', 'omega']
    _output_names = ['y']
    _slack_names = ['mu']
    _var_names_non_setable = ['delta', 'z']
    _var_names_controllable = ['u', 'delta', 'z', 'mu']

    _state_input_names = _state_names + _input_names
    _var_names = _state_input_names + _output_names + _slack_names
    _var_const_names = _state_input_names + [None] + _output_names + _slack_names  # const has no name.

    _var_to_dim_names_map = OrderedStructDict([(name, "".join(['n', name])) for name in _var_names])
    _var_to_const_dim_names_map = OrderedStructDict(
        [(name, "".join(['n', name]) if name else None) for name in _var_const_names])
    _var_to_bin_dim_names_map = OrderedStructDict([(name, "".join(['n', name, '_l'])) for name in _var_names])
    _var_to_type_names_map = OrderedStructDict([(name, "".join(['var_type_', name])) for name in _var_names])

    # Mld dimension mapping
    _mld_dim_map = {
        # Max row dimension of (*):
        'n_states': (0, ('A', 'B1', 'B2', 'B3', 'B4', 'b5')),
        'n_outputs': (0, ('C', 'D1', 'D2', 'D3', 'D4', 'd5')),
        'n_cons': (0, ('E', 'F1', 'F2', 'F2', 'F4', 'f5', 'G', 'Psi')),
        'nx': (0, ('A', 'B1', 'B2', 'B3', 'B4', 'b5')),
        'ny': (0, ('C', 'D1', 'D2', 'D3', 'D4', 'd5')),
        # Max column dimension of (*):
        'nu': (1, ('B1', 'D1', 'F1')),
        'ndelta': (1, ('B2', 'D2', 'F2')),
        'nz': (1, ('B3', 'D3', 'F3')),
        'nomega': (1, ('B4', 'D4', 'F4')),
        'nmu': (1, ('Psi',))
    }

    # MldInfo data layout
    _data_types = MldInfoDataTypes
    _data_layout = OrderedDict([(MldInfoDataTypes.sys_dims, _sys_dim_names),
                                (MldInfoDataTypes.state_input_dims,
                                 _var_to_dim_names_map.get_sub_list(_state_input_names)),
                                (MldInfoDataTypes.output_dim, _var_to_dim_names_map.get_sub_list(_output_names)),
                                (MldInfoDataTypes.slack_dim, _var_to_dim_names_map.get_sub_list(_slack_names)),
                                (MldInfoDataTypes.bin_dims, _var_to_bin_dim_names_map.get_sub_list(_var_names)),
                                (MldInfoDataTypes.var_types, _var_to_type_names_map.get_sub_list(_var_names)),
                                (MldInfoDataTypes.meta_data, _meta_data_names)])
    _allowed_data_set = set([data for data_type in _data_layout.values() for data in data_type])

    def __init__(self, mld_info_data=None, dt=None, param_struct=None, bin_dims_struct=None, var_types_struct=None,
                 required_params=None, **kwargs):
        super(MldInfo, self).__init__(**kwargs)

        self.update(mld_info_data=mld_info_data, dt=dt, param_struct=param_struct, bin_dims_struct=bin_dims_struct,
                    var_types_struct=var_types_struct, required_params=required_params, _from_init=True, **kwargs)

    @property
    def var_types_struct(self):
        return StructDict({var_name: self[self._var_to_type_names_map[var_name]] for var_name in self._var_names})

    @property
    def state_input_dims_struct(self):
        return StructDict(
            {var_name: self[self._var_to_dim_names_map[var_name]] for var_name in self._state_input_names})

    @property
    def var_dims_struct(self):
        return StructDict({var_name: self[self._var_to_dim_names_map[var_name]] for var_name in self._var_names})

    def get_var_type(self, var_name):
        return self[self._var_to_type_names_map[var_name]]

    def get_var_dim(self, var_name):
        return self[self._var_to_dim_names_map[var_name]]

    def get_var_bin_dim(self, var_name):
        return self[self._var_to_bin_dim_names_map[var_name]]

    @_process_mld_args_decor
    def update(self, mld_info_data=None, dt=None, param_struct=None, bin_dims_struct=None, var_types_struct=None,
               required_params=None, **kwargs):

        _from_init = kwargs.pop('_from_init', False)

        non_updateable = self._allowed_data_set.intersection(kwargs)
        if non_updateable:
            raise ValueError(
                "Cannot set values for keys: '{}' directly, these are updated automatically based on the MldModel "
                "structure and other valid parameters".format(non_updateable))
        elif kwargs:
            raise ValueError(
                "Invalid kwargs supplied: '{}'".format(kwargs)
            )

        mld_dims = None
        if mld_info_data is not None:
            if isinstance(mld_info_data, self.__class__):
                super(MldInfo, self).update(mld_info_data)
            elif isinstance(mld_info_data, dict):
                missing_keys = set(self._mld_dim_map).difference(mld_info_data)
                if missing_keys:
                    raise ValueError(
                        "If mld_info_data is a mld_dims struct, it requires and entry mld_dim in MldInfo._mld_dim_map, "
                        "missing these keys: {}".format(missing_keys))
                mld_dims = mld_info_data
            else:
                raise TypeError("Invalid type for mld_info_data.")

        if _from_init or any(item is not None for item in [dt, param_struct, required_params]):
            self._set_metadata(dt=dt, param_struct=param_struct, required_params=required_params)

        if mld_dims is not None:
            self._set_mld_dims(mld_dims)
        elif _from_init:
            mld_dims = {dim_name:(self[dim_name] or 0) for dim_name in self._mld_dim_map}
            self._set_mld_dims(mld_dims)

        if mld_dims or any(item is not None for item in [bin_dims_struct, var_types_struct]):
            self._set_var_info(bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct)

    def _set_metadata(self, dt=None, param_struct=None, required_params=None):
        dt_queue = [dt, self.get('dt')]
        dt = next((item for item in dt_queue if item is not None), 0)
        param_struct = param_struct or self['param_struct']
        required_params = required_params or self['required_params']
        super(MldInfo, self).update(dt=dt, param_struct=param_struct, required_params=required_params)

    def _set_var_info(self, bin_dims_struct=None, var_types_struct=None):
        # either set to new value, old value, or zero in that order - never None
        bin_dims_struct = bin_dims_struct or {}
        var_types_struct = var_types_struct or {}

        new_var_info = dict(self.items())

        for var_name in self._var_names:
            var_dim_name = self._var_to_dim_names_map[var_name]
            bin_dim_name = self._var_to_bin_dim_names_map[var_name]
            var_type_name = self._var_to_type_names_map[var_name]
            var_dim = self.get(var_dim_name)
            bin_dim = bin_dims_struct.get(bin_dim_name)
            var_type = var_types_struct.get(var_type_name)

            if var_name in self._var_names_non_setable:
                if bin_dim:
                    raise ValueError(
                        "Cannot manually set ndelta_l, nz_l - these are fixed by the MLD specification and dimension.")
                elif var_name == 'delta':
                    new_var_info[bin_dim_name] = bin_dim = self[var_dim_name]
                elif var_name == 'z':
                    new_var_info[bin_dim_name] = bin_dim = 0
            else:
                bin_dim_queue = [self._get_num_var_bin(var_type), bin_dim, self.get(bin_dim_name)]
                bin_dim = next((item for item in bin_dim_queue if item is not None), 0)
                new_var_info[bin_dim_name] = bin_dim

            if var_type is not None:
                var_type = self._validate_var_types_vect(var_type, new_var_info, bin_dim_name)
                if var_type.size != var_dim:
                    raise ValueError(
                        "Dimension of '{0}' must match dimension: '{1}'".format(var_type_name, var_dim_name))
                else:
                    new_var_info[var_type_name] = var_type
            else:
                try:
                    new_var_info[var_type_name] = np.vstack(
                        [np.repeat([['c']], var_dim - bin_dim, axis=0), np.repeat([['b']], bin_dim, axis=0)])
                except ValueError:
                    raise ValueError(
                        "Value of '{0}':{1} must be non-negative value <= dimension '{2}':{3}".format(
                            bin_dim_name, bin_dim, var_dim_name, var_dim))

        super(MldInfo, self).update(new_var_info)

    def _validate_var_types_vect(self, var_types_vect, mld_info_data, bin_dim_name=None):
        if var_types_vect is None:
            return var_types_vect
        else:
            var_types_vect = np.ravel(var_types_vect)[:, np.newaxis]
            if not np.setdiff1d(var_types_vect, self._valid_var_types).size == 0:
                raise ValueError('All elements of var_type_vectors must be in {}'.format(self._valid_var_types))
            if mld_info_data and bin_dim_name and (
                    mld_info_data.get(bin_dim_name) != self._get_num_var_bin(var_types_vect)):
                raise ValueError(
                    "Number of binary variables in var_type_vect:'{0}', does not match dimension of '{1}':{2}".format(
                        var_types_vect, bin_dim_name, mld_info_data.get(bin_dim_name)))
            return var_types_vect

    def _set_mld_dims(self, mld_dims):
        if mld_dims is not None:
            super(MldInfo, self).update(mld_dims)

    def _get_mld_dims(self, mld_mat_shapes_struct):
        mld_dims = StructDict()
        for dim_name, (axis, sys_matrix_ids) in self._mld_dim_map.items():
            shapes = StructDict.get_sub_list(mld_mat_shapes_struct, sys_matrix_ids)
            max_shape = _get_max_shape(shapes)
            mld_dims[dim_name] = max_shape[axis]
        return mld_dims

    @staticmethod
    def _get_num_var_bin(var_types_vect):
        if var_types_vect is None:
            return None
        else:
            var_types_vect_flat = np.ravel(var_types_vect)
            return (var_types_vect_flat == 'b').sum()


class MldModel(MldBase):
    __internal_names = ['_mld_info', '_mld_type', '_shapes_struct']
    _internal_names_set = MldBase._internal_names_set.union(__internal_names)

    _MLD_MODEL_TYPE_NAMES = ['numeric', 'symbolic', 'callable']
    _MldModelTypesNamedTup = NamedTuple('MldTypesNamedTup', _MLD_MODEL_TYPE_NAMES)
    MldModelTypes = _MldModelTypesNamedTup._make(_MLD_MODEL_TYPE_NAMES)

    _MLD_MODEL_MAT_TYPES_NAMES = ['state_input', 'output', 'constraint']
    _MldModelMatTypesNamedTup = NamedTuple('MldMatTypesNamedTup', _MLD_MODEL_MAT_TYPES_NAMES)
    MldModelMatTypes = _MldModelMatTypesNamedTup._make(_MLD_MODEL_MAT_TYPES_NAMES)

    ## Mld Model Config
    _state_input_mat_names = ['A', 'B1', 'B2', 'B3', 'B4', 'b5']
    _output_mat_names = ['C', 'D1', 'D2', 'D3', 'D4', 'd5']
    _constraint_mat_names = ['E', 'F1', 'F2', 'F3', 'F4', 'f5', 'G', 'Psi']
    _offset_vect_names = _MldModelMatTypesNamedTup('b5', 'd5', 'f5')

    _data_types = MldModelMatTypes
    _data_layout = OrderedDict([(MldModelMatTypes.state_input, _state_input_mat_names),
                                (MldModelMatTypes.output, _output_mat_names),
                                (MldModelMatTypes.constraint, _constraint_mat_names)])

    _allowed_data_set = set([data for data_type in _data_layout.values() for data in data_type])

    _sys_matrix_names_map =_MldModelMatTypesNamedTup(_state_input_mat_names, _output_mat_names, _constraint_mat_names)

    def __init__(self, system_matrices=None, dt=None, param_struct=None, bin_dims_struct=None, var_types_struct=None,
                 **kwargs):
        super(MldModel, self).__init__(**kwargs)
        super(MldModel, self).update(dict.fromkeys(self._allowed_data_set, np.empty(shape=(0, 0))))
        self._mld_info = MldInfo()
        self._mld_type = None
        self._shapes_struct = None

        self.update(system_matrices=system_matrices, dt=dt, param_struct=param_struct, bin_dims_struct=bin_dims_struct,
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

    @_process_mld_args_decor
    def update(self, system_matrices=None, dt=None, param_struct=None, bin_dims_struct=None, var_types_struct=None,
               **kwargs):
        from_init = kwargs.pop('from_init', False)
        mld_info_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in MldInfo._allowed_data_set}

        if system_matrices and kwargs:
            raise ValueError("Individual matrix arguments cannot be set if 'system_matrices' argument is set")
        elif system_matrices and isinstance(system_matrices, self.__class__):
            self._mld_info = system_matrices._mld_info
            self._mld_type = system_matrices._mld_type
            self._shapes_struct = system_matrices._shapes_struct
            super(MldModel, self).update(system_matrices)

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
                    if system_matrix is not None:
                        if isinstance(system_matrix, (sp.Expr)):
                            system_matrix = sp.Matrix([system_matrix])
                        elif not isinstance(system_matrix, sp.Matrix) and not callable(system_matrix):
                            system_matrix = np.atleast_2d(system_matrix)
                        new_sys_mats[sys_matrix_id] = system_matrix
                else:
                    if sys_matrix_id in kwargs:
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

        shapes_struct = _get_expr_shapes(new_sys_mats)
        if shapes_struct != self._shapes_struct:
            mld_dims = self._mld_info._get_mld_dims(shapes_struct)
            try:
                new_sys_mats, shapes_struct = self._validate_sys_matrix_shapes(new_sys_mats, shapes_struct, mld_dims)
            except ValueError as ve:
                raise ve
        else:
            mld_dims = None

        self._shapes_struct = shapes_struct
        self._set_mld_type(new_sys_mats)
        super(MldModel, self).update(new_sys_mats)

        if self.mld_type == self.MldModelTypes.symbolic:
            required_params = self._get_all_syms_str_list()
        else:
            required_params = None

        self.mld_info.update(mld_info_data=mld_dims, dt=dt, param_struct=param_struct, bin_dims_struct=bin_dims_struct,
                             var_types_struct=var_types_struct, required_params=required_params, **mld_info_kwargs)

    # TODO not finished - needs to include output constraints, interpolation and datetime handling
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
                    (self.E @ x_out[k, :] + self.F1 @ u[k, :] + self.F2 @ delta[k, :]
                     + self.F3 @ z[k, :] + self.F4 @ omega[k, :]) <= self.f5.T)

        x_out = x_out[:-1, :]  # remove last point for equal length output arrays.

        return OrderedStructDict(t_out=t_out, y_out=y_out, x_out=x_out, con_out=con_out)

    def to_numeric(self, param_struct=None, copy=True):
        if param_struct is None:
            param_struct = self.mld_info['param_struct'] or {}

        if self.mld_type == self.MldModelTypes.callable:
            numeric_dict = {mat_id: mat_eval(param_struct=param_struct) for mat_id, mat_eval in self.items()}
        elif self.mld_type == self.MldModelTypes.symbolic:
            print("Performance warning, mld_type is not callable had to convert to callable")
            eval_mld = self.to_eval()
            return eval_mld.to_numeric(param_struct=param_struct)
        else:
            numeric_dict = _deepcopy(dict(self)) if copy else dict(self)

        new_state = _deepcopy(self.__dict__) if copy else self.__dict__
        new_mld_info = new_state['_mld_info'] if copy else self.mld_info.copy()
        new_mld_info._sdict_setitem('param_struct', _copy(param_struct))

        return self.__class__._constructor(numeric_dict, new_state,
                                           attribute_override={'_mld_type': self.MldModelTypes.numeric,
                                                               '_mld_info': new_mld_info})

    def to_eval(self, copy=True):
        if self.mld_type in (self.MldModelTypes.numeric, self.MldModelTypes.symbolic):
            mld_eval_dict = {}
            lambda_cons = lambda x: x
            for mat_id, expr in self.items():
                if not isinstance(expr, (sp.Expr, sp.Matrix)):
                    lam = functools.partial(lambda_cons, np.atleast_2d(expr))
                    syms_str_tup = ()
                else:
                    if not isinstance(expr, sp.Matrix):
                        expr = sp.Matrix(np.atleast_2d(expr))
                    syms_tup, syms_str_tup = _get_syms_tup(expr)
                    lam = sp.lambdify(syms_tup, expr, modules="numpy", dummify=False)

                mld_eval_dict[mat_id] = _lambda_wrapper(lam, syms_str_tup, wrapped_name="".join([mat_id, "_eval"]))
        else:
            mld_eval_dict = _deepcopy(dict(self)) if copy else dict(self)

        new_state = _deepcopy(self.__dict__) if copy else self.__dict__
        new_mld_info = new_state['_mld_info'] if copy else self.mld_info.copy()
        return self.__class__._constructor(mld_eval_dict, new_state,
                                           attribute_override={'_mld_type': self.MldModelTypes.callable,
                                                               '_mld_info': new_mld_info})

    def _set_mld_type(self, mld_data):
        is_symbolic = [isinstance(sys_mat, (sp.Expr, sp.Matrix)) for sys_mat in mld_data.values()]
        if any(is_symbolic):
            self._mld_type = self.MldModelTypes.symbolic
        else:
            is_callable = [callable(sys_mat) for sys_mat in mld_data.values()]
            if any(is_callable):
                if not all(is_callable):
                    raise ValueError("If any mld matrices callable, all must be callable")
                else:
                    self._mld_type = self.MldModelTypes.callable
            else:
                self._mld_type = self.MldModelTypes.numeric

    def _validate_sys_matrix_shapes(self, mld_data, shapes_struct, mld_dims):
        A_shape = shapes_struct['A']
        C_shape = shapes_struct['C']

        if (A_shape[0] != A_shape[1]) and (0 not in A_shape):
            raise ValueError("Invalid shape for state matrix A:'{}', must be a square matrix or scalar".format(A_shape))

        shape_mat_dim_err_fmt = (
            "Invalid shape for {mat_type} matrix/vector '{mat_id}':{mat_shape}, must have same {dim_type} "
            "dimension as {req_mat_type} matrix '{req_mat_id}', i.e. '{req_shape}'").format

        row_fmt_str = ("({0}, *)").format
        col_fmt_str = ("(*, {0})").format

        def _validate_sys_matrix_shape(mat_type, mat_id, mat_shape, var_dim_name, sys_dim_name):
            var_dim = mld_dims.get(var_dim_name)
            sys_dim = mld_dims[sys_dim_name]
            if mat_type is self.MldModelMatTypes.state_input and mat_shape[0] != A_shape[0] and (0 not in A_shape):
                raise ValueError(
                    shape_mat_dim_err_fmt(mat_type=mat_type, mat_id=mat_id, mat_shape=mat_shape, dim_type='row',
                                          req_mat_type='state', req_mat_id='A',
                                          req_shape=row_fmt_str(A_shape[0])))
            elif mat_type is self.MldModelMatTypes.output and mat_shape[0] != C_shape[0] and (0 not in C_shape):
                raise ValueError(
                    shape_mat_dim_err_fmt(mat_type=mat_type, mat_id=mat_id, mat_shape=mat_shape, dim_type='row',
                                          req_mat_type='output', req_mat_id='C',
                                          req_shape=row_fmt_str(C_shape[0])))
            elif var_dim is not None and mat_shape[1] != var_dim:
                raise ValueError(
                    f"Invalid shape for {mat_type} matrix/vector '{mat_id}':{mat_shape}, column dimension must be "
                    f"equal to var dim '{var_dim_name}':{var_dim}, i.e. '{col_fmt_str(var_dim)}'")
            elif var_dim is None and mat_shape[1] != 1:
                raise ValueError(
                    f"'{mat_id}' must be of column vector, scalar or null array, currently has shape:{mat_shape}"
                )
            elif mat_shape[0] != sys_dim:
                raise ValueError(
                    f"Invalid shape for {mat_type} matrix/vector '{mat_id}':{mat_shape}, row dimension must be equal "
                    f"to system dimension - '{sys_dim_name}':{sys_dim}, i.e. '{row_fmt_str(sys_dim)}'. "
                    f"Note '{sys_dim_name}' is the maximum row dimension of all {mat_type} matrix/vectors.")

        def _gen_zero_empty_sys_mat(mat_type, mat_id, var_dim_name, sys_dim_name):
            var_dim = mld_dims.get(var_dim_name)
            sys_dim = mld_dims[sys_dim_name]

            if var_dim is None and mat_type is not self.MldModelMatTypes.constraint:
                var_dim = 1

            if var_dim and var_dim > 0:
                shape = (sys_dim, var_dim)
                return np.zeros(shape), shape
            else:
                shape = (sys_dim, 0)
                return mld_data[mat_id].reshape(shape), shape

        _var_const_dim_names = self._mld_info._var_to_const_dim_names_map.values()

        sys_dim_name_triad = self._MldModelMatTypesNamedTup._make(self._mld_info._sys_dim_names)
        sys_mat_ids_zip = zip_longest(self._state_input_mat_names, self._output_mat_names, self._constraint_mat_names)
        for var_dim_name, sys_mat_name_triad in zip_longest(_var_const_dim_names, sys_mat_ids_zip):
            for mat_type, mat_id, sys_dim_name in zip(self.MldModelMatTypes, sys_mat_name_triad, sys_dim_name_triad):
                if mat_id is None:
                    continue
                mat_shape = shapes_struct.get(mat_id)
                if 0 not in mat_shape:
                    _validate_sys_matrix_shape(mat_type, mat_id, mat_shape, var_dim_name, sys_dim_name)
                elif not callable(mld_data[mat_id]):
                    mld_data[mat_id], new_shape = _gen_zero_empty_sys_mat(mat_type, mat_id, var_dim_name, sys_dim_name)
                    shapes_struct[mat_id] = new_shape
                # todo add callable option

        if mld_dims[sys_dim_name_triad.constraint] and (0 in shapes_struct[self._offset_vect_names.constraint]):
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

        concat_var_type_info = StructDict.fromkeys(MldInfo._var_to_type_names_map)
        for var_type_name in concat_var_type_info:
            concat_info_list = []
            for model in mld_model_list:
                concat_info_list.append(model.mld_info[var_type_name])

            concat_var_type_info[var_type_name] = np.hstack(concat_info_list)

        concat_var_type_info.var_type_delta = None
        concat_var_type_info.var_type_z = None

        concat_mld = MldModel(concat_sys_mats, var_types_struct=concat_var_type_info)

        return concat_mld


def _as2darray(array):
    if array is None:
        out_array = np.empty(shape=(0, 0))
    else:
        out_array = np.array(array)
        if out_array.ndim == 1:  # return column array if 1 dim
            out_array = out_array[:, np.newaxis]
    return out_array


def _get_expr_shape(expr):
    try:
        expr_shape = expr.shape
    except AttributeError:
        pass
    else:
        if 0 in expr_shape:
            return (0, 0)
        elif len(expr_shape) == 1:
            return (expr_shape[0], 1)
        elif len(expr_shape) == 2:
            return expr_shape
        else:
            raise NotImplementedError("Maximum supported dimension is 2, got {}".format(len(expr_shape)))

    if expr is None:
        return (0, 0)
    elif np.isscalar(expr) or isinstance(expr, sp.Expr):
        return (1, 1)
    elif callable(expr):
        try:
            return _get_expr_shape(expr(_empty_call=True))
        except TypeError as te:
            raise te
    else:
        raise TypeError("Invalid expression type: '{0}', for expr: '{1!s}'".format(type(expr), expr))


def _get_max_shape(shapes):
    shapes = np.asarray(list(shapes), dtype=np.int)
    return tuple(np.maximum.reduce(shapes))


def _get_expr_shapes(*args, get_max_dim=False):
    if not args:
        return None

    if isinstance(args[0], dict):
        shape_dict = {expr_id: _get_expr_shape(expr) for expr_id, expr in args[0].items()}
        if get_max_dim:
            return _get_max_shape(shape_dict.values())
        else:
            return StructDict(shape_dict)
    else:
        shapes = [_get_expr_shape(arg) for arg in args]
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


# todo STILL NEEDS WORK
def _lambda_wrapper(func, local_syms_str_tup, wrapped_name=None):
    if wrapped_name:
        func.__qualname__ = wrapped_name
        func.__name__ = wrapped_name

    @functools.wraps(func)
    def wrapped(*args, param_struct=None, **kwargs):
        arg_list = args
        if kwargs.pop('_empty_call', False):
            arg_list = [np.NaN] * len(local_syms_str_tup)
        elif args and isinstance(args[0], dict) or isinstance(param_struct, dict):
            param_struct = args[0] if args else param_struct
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
    a.update(b5=np.ones((2, 1)), d5=np.ones((2, 1)), f5=np.ones((2, 1)))
    mld = MldModel(a)
    mld2 = MldModel({'A': 1})
    #
    # from models.model_generators import DewhModelGenerator
    # b = DewhModelGenerator().gen_dewh_symbolic_sys_matrices()
    # c = b.to_eval()
