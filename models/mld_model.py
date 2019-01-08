import functools
import inspect
import collections
from collections import OrderedDict, namedtuple as NamedTuple
from copy import deepcopy as _deepcopy, copy as _copy
from itertools import zip_longest

from operator import itemgetter as _itemgetter
from builtins import property as _property

from utils.func_utils import get_cached_func_spec, ParNotSet
from utils.matrix_utils import atleast_2d_col

import numpy as np
import scipy.linalg as scl
import scipy.sparse as scs
import sympy as sp
import wrapt

from utils.structdict import StructDict, OrderedStructDict, struct_repr, StructDictMeta


class _MldMeta(StructDictMeta):
    def __new__(cls, name, bases, _dict, **kwargs):
        kls = super(_MldMeta, cls).__new__(cls, name, bases, _dict, **kwargs)

        def _itemsetter(name):
            def caller(self, value):
                self.__setitem__(name, value)

            return caller

        for name in kls._allowed_data_set:
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
            raise KeyError("key: '{}' is not in self._allowed_data_set.".format(key))

    def _sdict_setitem(self, key, value):
        super(MldBase, self).__setitem__(key, value)

    def clear(self):
        self.__init__()

    def pop(self, *args, **kwargs):
        raise NotImplementedError("Items can not be removed from mld_model.")

    def popitem(self, *args, **kwargs):
        raise NotImplementedError("Items can not be removed from mld_model.")

    def get_sub_struct(self, keys, default=ParNotSet):
        return StructDict.sub_struct_fromdict(self, keys, default=default)

    def __delitem__(self, key):
        raise NotImplementedError("Items can not be removed from mld_model.")

    def __repr__(self):
        def value_repr(value): return (
            struct_repr(value, type_name='', repr_format_str='{type_name}{{{key_arg}{items}}}', align_values=True))

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
        state = {name: _copy(item) for name, item in self.__dict__.items()}
        return self.__class__._constructor(items=self.as_base_dict(), state=state)

    __copy__ = copy

    def deepcopy(self, memo=None):
        return self.__class__._constructor(items=self.as_base_dict(), state=self.__dict__, copy_items=True,
                                           copy_state=True,
                                           memo=memo)

    __deepcopy__ = deepcopy

    def __reduce__(self):
        return (self.__class__._constructor, (self.as_base_dict(), self.__dict__))


def _process_mld_args_decor(func):
    @wrapt.decorator
    def _wrapper(func, self, args, kwargs):
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

    return _wrapper(func)


class MldInfo(MldBase):
    __internal_names = []
    _internal_names_set = MldBase._internal_names_set.union(__internal_names)

    _MLD_INFO_DATA_TYPE_NAMES = ['sys_dims', 'state_input_dims', 'output_dim', 'slack_dim', 'bin_dims', 'var_types',
                                 'meta_data']
    _MldInfoDataTypesNamedTup = NamedTuple('MldInfoTypesNamedTup', _MLD_INFO_DATA_TYPE_NAMES)
    MldInfoDataTypes = _MldInfoDataTypesNamedTup._make(_MLD_INFO_DATA_TYPE_NAMES)

    # MldInfo Config
    _valid_var_types = {'c', 'b'}
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
    _var_to_bin_dim_names_map = OrderedStructDict([(name, "".join(['n', name, '_l'])) for name in _var_names])
    _var_to_type_names_map = OrderedStructDict([(name, "".join(['var_type_', name])) for name in _var_names])

    _var_to_const_dim_names_map = OrderedStructDict(
        [(name, "".join(['n', name]) if name else None) for name in _var_const_names])

    _var_info_to_var_names_map = OrderedStructDict.combine_structs(_var_to_dim_names_map.to_reversed_map(),
                                                                   _var_to_bin_dim_names_map.to_reversed_map(),
                                                                   _var_to_type_names_map.to_reversed_map())

    # Mld dimension mapping
    _mld_dim_map = {
        # Max row dimension of (*):
        'n_states' : (0, ('A', 'B1', 'B2', 'B3', 'B4', 'b5')),
        'n_outputs': (0, ('C', 'D1', 'D2', 'D3', 'D4', 'd5')),
        'n_cons'   : (0, ('E', 'F1', 'F2', 'F2', 'F4', 'f5', 'G', 'Psi')),
        'nx'       : (0, ('A', 'B1', 'B2', 'B3', 'B4', 'b5')),
        'ny'       : (0, ('C', 'D1', 'D2', 'D3', 'D4', 'd5')),
        # Max column dimension of (*):
        'nu'       : (1, ('B1', 'D1', 'F1')),
        'ndelta'   : (1, ('B2', 'D2', 'F2')),
        'nz'       : (1, ('B3', 'D3', 'F3')),
        'nomega'   : (1, ('B4', 'D4', 'F4')),
        'nmu'      : (1, ('Psi',))
    }

    # MldInfo data layout
    _data_types = MldInfoDataTypes
    _data_layout = (
        OrderedDict([(MldInfoDataTypes.sys_dims, _sys_dim_names),
                     (MldInfoDataTypes.state_input_dims, _var_to_dim_names_map.get_sub_list(_state_input_names)),
                     (MldInfoDataTypes.output_dim, _var_to_dim_names_map.get_sub_list(_output_names)),
                     (MldInfoDataTypes.slack_dim, _var_to_dim_names_map.get_sub_list(_slack_names)),
                     (MldInfoDataTypes.bin_dims, _var_to_bin_dim_names_map.get_sub_list(_var_names)),
                     (MldInfoDataTypes.var_types, _var_to_type_names_map.get_sub_list(_var_names)),
                     (MldInfoDataTypes.meta_data, _meta_data_names)]))
    _allowed_data_set = set([data for data_type in _data_layout.values() for data in data_type])

    def __init__(self, mld_info_data=None, dt=None, param_struct=None, bin_dims_struct=None, var_types_struct=None,
                 required_params=None, **kwargs):
        super(MldInfo, self).__init__(**kwargs)

        self.update(mld_info_data=mld_info_data, dt=dt, param_struct=param_struct, bin_dims_struct=bin_dims_struct,
                    var_types_struct=var_types_struct, required_params=required_params, _from_init=True, **kwargs)

    def copy(self):
        items = self.as_base_dict()
        items['param_struct'] = _copy(items['param_struct'])
        return self.__class__._constructor(items, self.__dict__)

    __copy__ = copy

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
        bin_dims_struct = bin_dims_struct or {}
        var_types_struct = var_types_struct or {}

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

        new_mld_info = self.as_base_dict()

        if _from_init:
            if mld_dims is None:
                mld_dims = {dim_name: 0 for dim_name in self._mld_dim_map}
            if dt is None:
                dt = 0

        if mld_dims:
            new_mld_info.update(mld_dims)
            modified_var_names = self._var_names
        else:
            modified_var_names = set(
                [self._var_info_to_var_names_map[key] for key in set(bin_dims_struct) | set(var_types_struct)])

        new_mld_info = self._set_metadata(info_data=new_mld_info, dt=dt, param_struct=param_struct,
                                          required_params=required_params)

        new_mld_info = self._set_var_info(info_data=new_mld_info, modified_var_names=modified_var_names,
                                          bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct)

        super(MldInfo, self).update(new_mld_info)

    def _set_metadata(self, info_data, dt=None, param_struct=None, required_params=None):

        if dt is not None:
            info_data['dt'] = dt

        if param_struct is not None:
            if isinstance(param_struct, dict):
                info_data['param_struct'] = param_struct
            else:
                raise TypeError("'param_struct' must be dictionary like or None.")

        if required_params is not None:
            if isinstance(required_params, collections.Container):
                info_data['required_params'] = required_params
            else:
                raise TypeError("'required_params' must be subtype of collections.Container or None.")

        return info_data

    def _set_var_info(self, info_data, modified_var_names, bin_dims_struct, var_types_struct):

        for var_name in modified_var_names:
            var_dim_name = self._var_to_dim_names_map[var_name]
            bin_dim_name = self._var_to_bin_dim_names_map[var_name]
            var_type_name = self._var_to_type_names_map[var_name]
            var_dim = info_data.get(var_dim_name)
            bin_dim = bin_dims_struct.get(bin_dim_name)
            var_type = var_types_struct.get(var_type_name)

            if var_name in self._var_names_non_setable:
                if bin_dim:
                    raise ValueError(
                        "Cannot manually set ndelta_l, nz_l - these are fixed by the MLD specification and "
                        "dimension.")
                elif var_name == 'delta':
                    info_data[bin_dim_name] = bin_dim = info_data[var_dim_name]
                elif var_name == 'z':
                    info_data[bin_dim_name] = bin_dim = 0
            else:
                bin_dim_queue = [self._get_num_var_bin(var_type), bin_dim, info_data.get(bin_dim_name)]
                bin_dim = next((item for item in bin_dim_queue if item is not None), 0)
                info_data[bin_dim_name] = bin_dim

            if var_type is not None:
                var_type = self._validate_var_types_vect(var_type, info_data, bin_dim_name)
                if var_type.size != var_dim:
                    raise ValueError(
                        "Dimension of '{0}' must match dimension: '{1}'".format(var_type_name, var_dim_name))
                else:
                    info_data[var_type_name] = var_type
            else:
                if bin_dim > var_dim:
                    raise ValueError((f"Value of '{bin_dim_name}':{bin_dim} must be non-negative value <= "
                                      f"dimension '{var_dim_name}':{var_dim}"))
                info_data[var_type_name] = atleast_2d_col(list('c' * (var_dim - bin_dim) + 'b' * bin_dim))

        return info_data

    def _validate_var_types_vect(self, var_types_vect, mld_info_data, bin_dim_name=None):
        if var_types_vect is None:
            return var_types_vect
        else:
            var_types_vect = atleast_2d_col(var_types_vect, dtype=np.str)
            if not np.setdiff1d(var_types_vect, self._valid_var_types).size == 0:
                raise ValueError(f"All elements of var_type_vectors must be in {self._valid_var_types}")
            if mld_info_data and bin_dim_name and (
                    mld_info_data.get(bin_dim_name) != self._get_num_var_bin(var_types_vect)):
                raise ValueError(
                    "Number of binary variables in var_type_vect:'{0}', does not match dimension of '{1}':{2}".format(
                        var_types_vect, bin_dim_name, mld_info_data.get(bin_dim_name)))
            return var_types_vect

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
            return (np.asanyarray(var_types_vect, dtype=np.str) == 'b').sum()


class MldModel(MldBase):
    __internal_names = ['_mld_info', '_mld_type', '_shapes_struct']
    _internal_names_set = MldBase._internal_names_set.union(__internal_names)

    _MLD_MODEL_TYPE_NAMES = ['numeric', 'callable', 'symbolic']
    MldModelTypesNamedTup = NamedTuple('MldModelTypes', _MLD_MODEL_TYPE_NAMES)
    MldModelTypes = MldModelTypesNamedTup._make(_MLD_MODEL_TYPE_NAMES)

    _MLD_MODEL_MAT_TYPES_NAMES = ['state_input', 'output', 'constraint']
    MldModelMatTypesNamedTup = NamedTuple('MldMatTypes', _MLD_MODEL_MAT_TYPES_NAMES)
    MldModelMatTypes = MldModelMatTypesNamedTup._make(_MLD_MODEL_MAT_TYPES_NAMES)

    ## Mld Model Config
    _state_input_mat_names = ['A', 'B1', 'B2', 'B3', 'B4', 'b5']
    _output_mat_names = ['C', 'D1', 'D2', 'D3', 'D4', 'd5']
    _constraint_mat_names = ['E', 'F1', 'F2', 'F3', 'F4', 'f5', 'G', 'Psi']
    _offset_vect_names = MldModelMatTypesNamedTup('b5', 'd5', 'f5')

    _data_types = MldModelMatTypes
    _data_layout = OrderedDict([(MldModelMatTypes.state_input, _state_input_mat_names),
                                (MldModelMatTypes.output, _output_mat_names),
                                (MldModelMatTypes.constraint, _constraint_mat_names)])

    _allowed_data_set = set([data for data_type in _data_layout.values() for data in data_type])

    _sys_matrix_names_map = MldModelMatTypesNamedTup(_state_input_mat_names, _output_mat_names, _constraint_mat_names)

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

    def __repr__(self):
        base_repr = super(MldModel, self).__repr__()
        mld_model_str = ("\n"
                         "x(k+1) = A*x(k) + B1*u(k) + B2*delta(k) + B3*z(k) + B4*omega(k) + b5\n"
                         "y(k) = C*x(k) + D1*u(k) + D2*delta(k) + D3*z(k) + D4*omega(k) + d5\n"
                         "E*x(k) + F1*u(k) + F2*delta(k) + F3*z(k) + F4*omega(k) + G*y(k) <= f5 + Psi*mu(k)\n"
                         f"mld_type : {self.mld_type}\n"
                         "\n"
                         "with:\n")
        mld_model_repr = base_repr.replace('\n', '\n' + mld_model_str, 1)
        return mld_model_repr

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

        new_sys_mats = self.as_base_dict()
        try:
            for sys_matrix_id, system_matrix in creation_matrices.items():
                if sys_matrix_id in self._allowed_data_set:
                    if system_matrix is not None:
                        if isinstance(system_matrix, (sp.Expr, sp.Matrix)):  # it is symbolic
                            if isinstance(system_matrix, sp.Expr):
                                system_matrix = sp.Matrix([system_matrix])
                        elif callable(system_matrix):  # it is callable
                            system_matrix = CallableMatrix(system_matrix, sys_matrix_id)
                        else:  # it must be numeric
                            system_matrix = atleast_2d_col(system_matrix)
                            system_matrix.setflags(write=False)
                            if not np.issubdtype(system_matrix.dtype, np.number):
                                raise TypeError("System matrices must be numeric, callable, or symbolic.")
                        new_sys_mats[sys_matrix_id] = system_matrix
                else:
                    if sys_matrix_id in kwargs:
                        raise ValueError("Invalid matrix name in kwargs: {}".format(sys_matrix_id))
                    else:
                        raise ValueError("Invalid matrix name in system_matrices: {}".format(sys_matrix_id))
        except AttributeError as ae:
            raise TypeError(f"Argument:'system_matrices' must be dictionary like: {ae.args[0]}")

        if from_init and creation_matrices.get('C') is None:
            new_sys_mats['C'] = np.eye(*(_get_expr_shape(new_sys_mats['A'])), dtype=np.int)
            new_sys_mats['C'].setflags(write=False)

        shapes_struct = _get_expr_shapes(new_sys_mats)
        if shapes_struct != self._shapes_struct:
            mld_dims = self._mld_info._get_mld_dims(shapes_struct)
            new_sys_mats, shapes_struct = self._validate_sys_matrix_shapes(new_sys_mats, shapes_struct, mld_dims)
        else:
            mld_dims = None

        self._shapes_struct = shapes_struct
        new_sys_mats = self._set_mld_type(new_sys_mats)
        super(MldModel, self).update(new_sys_mats)

        if self.mld_type in (self.MldModelTypes.callable, self.MldModelTypes.symbolic):
            required_params = self._get_required_params()
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
            dict_numeric = {mat_id: mat_callable(param_struct=param_struct) for mat_id, mat_callable in self.items()}
        elif self.mld_type == self.MldModelTypes.symbolic:
            print("Performance warning, mld_type is not callable had to convert to callable")
            callable_mld = self.to_callable()
            return callable_mld.to_numeric(param_struct=param_struct)
        else:
            dict_numeric = _deepcopy(dict(self)) if copy else dict(self)

        # make all system matrices read_only
        for mat in dict_numeric.values():
            mat.setflags(write=False)

        new_state = _deepcopy(self.__dict__) if copy else {name: _copy(item) for name, item in self.__dict__.items()}
        new_state['_mld_info']._sdict_setitem('param_struct', _copy(param_struct))
        return self.__class__._constructor(dict_numeric, new_state,
                                           attribute_override={'_mld_type': self.MldModelTypes.numeric})

    def to_callable(self, copy=True):
        if self.mld_type in (self.MldModelTypes.numeric, self.MldModelTypes.symbolic):
            dict_callable = {}
            for sys_matrix_id, system_matrix in self.items():
                dict_callable[sys_matrix_id] = CallableMatrix(system_matrix, sys_matrix_id)
        else:
            dict_callable = _deepcopy(dict(self)) if copy else dict(self)

        new_state = _deepcopy(self.__dict__) if copy else {name: _copy(item) for name, item in self.__dict__.items()}
        return self.__class__._constructor(dict_callable, new_state,
                                           attribute_override={'_mld_type': self.MldModelTypes.callable})

    def _set_mld_type(self, mld_data):
        is_symbolic = {sys_mat_id: isinstance(sys_mat, (sp.Expr, sp.Matrix)) for sys_mat_id, sys_mat in
                       mld_data.items()}
        if any(is_symbolic.values()):
            self._mld_type = self.MldModelTypes.symbolic
        else:
            is_callable = {sys_mat_id: callable(sys_mat) for sys_mat_id, sys_mat in
                           mld_data.items()}
            if any(is_callable.values()):
                for sys_mat_id, is_mat_callable in is_callable.items():
                    if not is_mat_callable:
                        mld_data[sys_mat_id] = CallableMatrix(mld_data[sys_mat_id], sys_mat_id)
                self._mld_type = self.MldModelTypes.callable
            else:
                self._mld_type = self.MldModelTypes.numeric

        return mld_data

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
                new_shape = (sys_dim, var_dim)
                zero_empty_mat = np.zeros(new_shape, dtype=np.int)
            else:
                new_shape = (sys_dim, 0)
                zero_empty_mat = np.empty(shape=new_shape)

            if callable(mld_data[mat_id]):
                zero_empty_mat = CallableMatrix(zero_empty_mat, mat_id)

            return zero_empty_mat, new_shape

        _var_const_dim_names = self._mld_info._var_to_const_dim_names_map.values()

        sys_dim_name_triad = self.MldModelMatTypesNamedTup._make(self._mld_info._sys_dim_names)
        sys_mat_ids_zip = zip_longest(self._state_input_mat_names, self._output_mat_names, self._constraint_mat_names)
        for var_dim_name, sys_mat_name_triad in zip_longest(_var_const_dim_names, sys_mat_ids_zip):
            for mat_type, mat_id, sys_dim_name in zip(self.MldModelMatTypes, sys_mat_name_triad, sys_dim_name_triad):
                if mat_id is None:
                    continue
                mat_shape = shapes_struct.get(mat_id)
                if 0 not in mat_shape:
                    _validate_sys_matrix_shape(mat_type, mat_id, mat_shape, var_dim_name, sys_dim_name)
                else:
                    mld_data[mat_id], new_shape = _gen_zero_empty_sys_mat(mat_type, mat_id, var_dim_name, sys_dim_name)
                    shapes_struct[mat_id] = new_shape

        if mld_dims[sys_dim_name_triad.constraint] and (0 in shapes_struct[self._offset_vect_names.constraint]):
            raise ValueError(
                "Constraint vector '{}' can only be null if all constraint matrices are null.".format(
                    self._offset_vect_names.constraint)
            )

        return mld_data, shapes_struct

    def _get_required_params(self):
        str_list = []
        for sys_matrix in self.values():
            if isinstance(sys_matrix, (sp.Expr, sp.Matrix)):
                str_list.extend([str(sym) for sym in sys_matrix.free_symbols])
            elif isinstance(sys_matrix, CallableMatrix):
                str_list.extend(sys_matrix.required_params)
            elif callable(sys_matrix):
                str_list.extend(set(get_cached_func_spec(sys_matrix).all_kw_params).difference(['param_struct']))
        return sorted(set(str_list))

    # TODO REWORK
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


class CallableMatrix(wrapt.decorators.AdapterWrapper):

    def __init__(self, matrix, matrix_name=None):
        if isinstance(matrix, type(self)):
            super(CallableMatrix, self).__init__(wrapped=matrix.__wrapped__, wrapper=matrix._self_wrapper, enabled=None,
                                                 adapter=matrix._self_adapter)
        else:
            # used to wrap constant as a function
            def const_func(constant):
                def functor():
                    return constant

                return functor

            if isinstance(matrix, (sp.Expr, sp.Matrix)):
                system_matrix = sp.Matrix(matrix)
                param_sym_tup = _get_param_sym_tup(system_matrix)
                func = sp.lambdify(param_sym_tup, system_matrix, modules="numpy", dummify=False)
            elif callable(matrix):
                func = matrix
            else:
                func = const_func(atleast_2d_col(matrix))

            self._self_matrix_id = matrix_name
            self._self_orig_wrapped_name = func.__name__
            func.__name__ = self._self_matrix_id if matrix_name is not None else func.__name__
            func.__qualname__ = "".join(func.__qualname__.rsplit('.', 1)[:-1] + ['.', func.__name__]).lstrip('.')

            self._self_wrapped_f_spec = get_cached_func_spec(func, save_to_cache=False, bypass_cache=True)
            adapter = self._gen_adapter()
            self._self_adapter_spec = get_cached_func_spec(adapter, save_to_cache=False, bypass_cache=True)
            wrapper = self._matrix_wrapper

            super(CallableMatrix, self).__init__(wrapped=func, wrapper=wrapper, enabled=None, adapter=adapter)

        try:
            self.__delattr__('_self_shape')
        except AttributeError:
            pass

        # get relevant array attributes by performing a function call with all arguments set to NaN
        try:
            nan_call = self._nan_call()
        except TypeError:
            msg = f"Cannot determine shape of callable, it is likely not constant.\n"
            note = (
                "Note: all callable expressions must return with a constant array shape. Shape is determined by "
                "calling the function with all arguments set to a float with value NaN.")
            raise TypeError(msg + note)

        self._self_shape = _get_expr_shape(nan_call)
        self._self_size = np.prod(nan_call.size)
        self._self_ndim = nan_call.ndim
        self._self_dtype = nan_call.dtype
        self._self_nbytes = nan_call.nbytes
        self._self_itemsize = nan_call.itemsize

    def _nan_call(self):
        kwargs = {param_name: np.NaN for param_name in self._self_wrapped_f_spec.all_kw_params}
        args = [np.NaN] * len(self._self_wrapped_f_spec.pos_only_params)
        return self(*args, **kwargs)

    def _matrix_wrapper(self, wrapped, instance, args, kwargs):
        param_struct = kwargs.pop('param_struct', None)
        if param_struct and self._self_wrapped_f_spec.all_kw_params:
            try:
                duplicates = set(kwargs).intersection(param_struct) if kwargs else None
                kwargs.update(
                    {name: param_struct[name] for name in
                     set(self._self_wrapped_f_spec.all_kw_params).intersection(param_struct)})
            except TypeError as te:
                msg = f"'param_struct' must be dictionary like or None: {te.args[0]}"
                raise TypeError(msg).with_traceback(te.__traceback__) from None
            else:
                if duplicates:
                    raise TypeError(
                        f"{wrapped.__name__}() got multiple values for argument '{duplicates.pop()}' - values in "
                        f"kwargs are duplicated in param_struct.")

        try:
            retval = wrapped(*args, **kwargs)
        except TypeError as te:
            msg = te.args[0].replace(self._self_orig_wrapped_name, wrapped.__name__)
            raise TypeError(msg).with_traceback(te.__traceback__) from None

        if getattr(retval, 'ndim', 0) < 2:
            retval = atleast_2d_col(retval)

        retval.setflags(write=False)
        return retval

    def _gen_adapter(self):
        f_args_spec_struct = OrderedStructDict(self._self_wrapped_f_spec.arg_spec._asdict()).deepcopy()
        f_args_spec_struct.kwonlyargs.append('param_struct')
        if f_args_spec_struct.kwonlydefaults:
            f_args_spec_struct.kwonlydefaults.update({'param_struct': None})
        else:
            f_args_spec_struct.kwonlydefaults = {'param_struct': None}

        f_args_spec = inspect.FullArgSpec(**f_args_spec_struct)
        adapter = inspect.formatargspec(*f_args_spec)
        ns = {}
        exec('def adapter{0}: pass'.format(adapter), ns, ns)
        adapter = ns['adapter']
        return adapter

    def __reduce__(self):
        return (type(self), (self.__wrapped__, self._self_matrix_id))

    @property
    def _f_spec(self):
        return self._self_adapter_spec

    @_f_spec.setter
    def _f_spec(self, f_spec):
        self._self_adapter_spec = f_spec

    @property
    def required_params(self):
        return self._self_wrapped_f_spec.all_kw_params

    @property
    def matrix_id(self):
        return self._self_matrix_id

    @property
    def shape(self):
        return self._self_shape

    @property
    def size(self):
        return self._self_size

    @property
    def ndim(self):
        return self._self_ndim

    @property
    def dtype(self):
        return self._self_dtype

    @property
    def nbytes(self):
        return self._self_nbytes

    @property
    def itemsize(self):
        return self._self_itemsize

    def __repr__(self):
        empty_str = f", shape={self._self_shape}" if not self._self_size else ""
        return f"CallableMatrix{self.__signature__}{empty_str}"

    def __str__(self):
        return self.__repr__()

    def __dir__(self):
        wrapped_dir = set(dir(self.__wrapped__))
        added_dir = set(type(self).__dict__)
        rv = wrapped_dir | added_dir
        return sorted(rv)


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
        if len(expr_shape) <= 2:
            return expr_shape
        else:
            raise NotImplementedError("Maximum supported dimension is 2, got {}".format(len(expr_shape)))

    if expr is None:
        return (0, 0)
    elif np.isscalar(expr) or isinstance(expr, sp.Expr):
        return (1, 1)
    elif callable(expr):
        expr = CallableMatrix(expr)
        return expr.shape
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

def _get_param_sym_tup(expr):
    try:
        sym_dict = {str(sym): sym for sym in expr.free_symbols}
        param_sym_tup = tuple([sym_dict.get(sym) for sym in sorted(sym_dict.keys())])
    except AttributeError:
        param_sym_tup = ()

    return param_sym_tup


if __name__ == '__main__':
    a = dict.fromkeys(MldModel._allowed_data_set, np.ones((2, 2)))
    a.update(b5=np.ones((2, 1)), d5=np.ones((2, 1)), f5=np.ones((2, 1)))
    mld = MldModel(a)
    mld2 = MldModel({'A': 1})
    #
    # from models.model_generators import DewhModelGenerator
    # b = DewhModelGenerator().gen_dewh_symbolic_sys_matrices()
    # c = b.to_callable()
