import collections
from collections import OrderedDict, namedtuple as NamedTuple
from copy import deepcopy as _deepcopy, copy as _copy
from reprlib import recursive_repr as _recursive_repr

from utils.decorator_utils import process_method_args_decor
from utils.func_utils import get_cached_func_spec, ParNotSet
from utils.helper_funcs import num_not_None, is_all_None, is_any_None
from utils.matrix_utils import atleast_2d_col, CallableMatrix, get_expr_shape, get_expr_shapes, matmul

import numpy as np
import scipy.linalg as scl
import scipy.sparse as scs
import sympy as sp
import wrapt

import cvxpy as cvx
from cvxpy.expressions import expression as cvx_e

from structdict import StructDict, OrderedStructDict, struct_repr, StructPropDictMixin, struct_prop_dict

from utils.versioning import VersionMixin, increments_version_decor, versioned


@versioned
class MldBase(StructPropDictMixin, dict):
    __internal_names = ()

    _data_types = ()
    _data_layout = {}
    _field_names = ()
    _field_names_set = frozenset(_field_names)

    @increments_version_decor
    def __init__(self, *args, **kwargs):
        super(MldBase, self).__init__()
        super(MldBase, self).update(dict.fromkeys(self._field_names_set))

    @increments_version_decor
    def __setitem__(self, key, value):
        if key in self._field_names_set:
            self.update(**{key: value})
        else:
            raise KeyError("key: '{}' is not in self._field_names_set.".format(key))

    @increments_version_decor
    def clear(self):
        self.__init__()

    def pop(self, *args, **kwargs):
        raise NotImplementedError(f"Items cannot be removed from a {self.__class__.__name__} object.")

    def popitem(self, *args, **kwargs):
        raise NotImplementedError(f"Items cannot be removed from a {self.__class__.__name__} object.")

    @increments_version_decor
    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self.__setitem__(key, default)
            return default

    def get_sub_struct(self, keys, default=ParNotSet):
        return StructDict.sub_struct_fromdict(self, keys, default=default)

    def __delitem__(self, key):
        raise NotImplementedError(f"Items cannot be removed from a {self.__class__.__name__} object.")

    def __repr__(self):
        def value_repr(value): return (
            struct_repr(value, type_name='', repr_format_str='{type_name}{{{key_arg}{items}}}', align_values=True))

        repr_dict = {data_type: value_repr(self.get_sub_struct(self._data_layout[data_type])) for data_type in
                     self._data_types}
        return struct_repr(repr_dict, type_name=self.__class__.__name__, align_values=True, align_padding_width=1,
                           value_format_str='\b{value}')

    def copy(self):
        return self._constructor_from_self(copy_instance_attr=True)

    __copy__ = copy


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


_shape_func_map = NamedTuple('shape_func_map', ['axis', 'func', 'items'])


@versioned(versioned_sub_objects=())
class MldInfo(MldBase):
    __internal_names = ()
    __slots__ = ()

    _MLD_INFO_DATA_TYPE_NAMES = ['sys_dims', 'var_dims',
                                 'bin_dims', 'var_types', 'meta_data']
    _MldInfoDataTypesNamedTup = NamedTuple('MldInfoTypesNamedTup', _MLD_INFO_DATA_TYPE_NAMES)
    MldInfoDataTypes = _MldInfoDataTypesNamedTup._make(_MLD_INFO_DATA_TYPE_NAMES)

    # MldInfo Config
    _valid_var_types = {'c', 'b'}
    _meta_data_names = ['dt', 'param_struct', 'required_params']

    _sys_dim_names = ['n_states', 'n_outputs', 'n_constraints']

    _state_var_names = ['x']
    _input_var_names = ['u', 'delta', 'z', 'omega']
    _output_var_names = ['y']
    _slack_var_names = ['mu']
    _non_setable_var_names = ['delta', 'z', 'v']
    _controllable_var_names = ['u', 'delta', 'z', 'mu']
    _concat_controllable_var_names = ['v']

    _state_input_var_names = _state_var_names + _input_var_names
    _var_names = (_state_input_var_names + _output_var_names + _slack_var_names +
                  _concat_controllable_var_names)
    _var_and_const_names = _state_input_var_names + ['constant'] + _output_var_names + _slack_var_names

    _var_to_dim_names_map = OrderedStructDict([(name, "".join(['n', name])) for name in _var_names])
    _var_to_bin_dim_names_map = OrderedStructDict([(name, "".join(['n', name, '_l'])) for name in _var_names])
    _var_to_type_names_map = OrderedStructDict([(name, "".join(['var_type_', name])) for name in _var_names])

    _var_and_const_to_dim_names_map = OrderedStructDict(
        [(name, "".join(['n', name]) if name != 'constant' else None) for name in _var_and_const_names])

    _var_info_to_var_names_map = OrderedStructDict.combine_structs(_var_to_dim_names_map.to_reversed_map(),
                                                                   _var_to_bin_dim_names_map.to_reversed_map(),
                                                                   _var_to_type_names_map.to_reversed_map())

    # Mld dimension mapping
    _mld_dim_map = {
        # Max row dimension of (*):
        'n_states'     : (np.maximum.reduce, 0, ['A', 'B1', 'B2', 'B3', 'B4', 'b5']),
        'n_outputs'    : (np.maximum.reduce, 0, ['C', 'D1', 'D2', 'D3', 'D4', 'd5']),
        'n_constraints': (np.maximum.reduce, 0, ['E', 'F1', 'F2', 'F3', 'F4', 'f5', 'G', 'Psi']),
        'nx'           : (np.maximum.reduce, 0, ['A', 'B1', 'B2', 'B3', 'B4', 'b5']),
        'ny'           : (np.maximum.reduce, 0, ['C', 'D1', 'D2', 'D3', 'D4', 'd5']),
        # Max column dimension of (*):
        'nu'           : (np.maximum.reduce, 1, ['B1', 'D1', 'F1']),
        'ndelta'       : (np.maximum.reduce, 1, ['B2', 'D2', 'F2']),
        'nz'           : (np.maximum.reduce, 1, ['B3', 'D3', 'F3']),
        'nomega'       : (np.maximum.reduce, 1, ['B4', 'D4', 'F4']),
        'nmu'          : (np.maximum.reduce, 1, ['Psi']),
        # Add dims of (*):
        'nv'           : (np.maximum.reduce, 0,
                          [(np.add.reduce, 1, ['B1', 'B2', 'B3']),
                           (np.add.reduce, 1, ['D1', 'D2', 'D3']),
                           (np.add.reduce, 1, ['F1', 'F2', 'F3', 'Psi'])])

    }

    # MldInfo data layout
    _data_types = MldInfoDataTypes
    _data_layout = (
        OrderedDict([(MldInfoDataTypes.sys_dims, _sys_dim_names),
                     (MldInfoDataTypes.var_dims, _var_to_dim_names_map.get_sub_list(_var_names)),
                     (MldInfoDataTypes.bin_dims, _var_to_bin_dim_names_map.get_sub_list(_var_names)),
                     (MldInfoDataTypes.var_types, _var_to_type_names_map.get_sub_list(_var_names)),
                     (MldInfoDataTypes.meta_data, _meta_data_names)]))
    _field_names = sorted([data for data_type in _data_layout.values() for data in data_type])
    _field_names_set = frozenset(_field_names)

    def __init__(self, mld_info_data=None, dt=ParNotSet, param_struct=None, bin_dims_struct=None, var_types_struct=None,
                 required_params=None, **kwargs):
        super(MldInfo, self).__init__(**kwargs)

        self.update(mld_info_data=mld_info_data, dt=dt, param_struct=param_struct, bin_dims_struct=bin_dims_struct,
                    var_types_struct=var_types_struct, required_params=required_params, _from_init=True, **kwargs)

    def copy(self):
        param_struct_copy = _copy(self['param_struct'])
        return self._constructor_from_self(items_override={'param_struct': param_struct_copy}, copy_instance_attr=True)

    __copy__ = copy

    VarTypesStruct = struct_prop_dict('VarTypesStruct', _var_names, sorted_repr=False)
    VarDimStruct = struct_prop_dict('VarDimStruct', _var_names, sorted_repr=False)
    VarBinDimStruct = struct_prop_dict('VarBinDimStruct', _var_names, sorted_repr=False)

    @property
    def var_types_struct(self):
        return self.VarTypesStruct({var_name: self.get_var_type(var_name) for var_name in self._var_names})

    @property
    def var_dims_struct(self):
        return self.VarDimStruct({var_name: self.get_var_dim(var_name) for var_name in self._var_names})

    @property
    def var_bin_dims_struct(self):
        return self.VarBinDimStruct({var_name: self.get_var_bin_dim(var_name) for var_name in self._var_names})

    def get_var_type(self, var_name):
        return self[self._var_to_type_names_map[var_name]]

    def get_var_dim(self, var_name):
        return self[self._var_to_dim_names_map[var_name]]

    def get_var_bin_dim(self, var_name):
        return self[self._var_to_bin_dim_names_map[var_name]]

    @_process_mld_args_decor
    @increments_version_decor
    def update(self, mld_info_data=None, dt=ParNotSet, param_struct=None, bin_dims_struct=None, var_types_struct=None,
               required_params=None, **kwargs):

        _from_init = kwargs.pop('_from_init', False)
        bin_dims_struct = bin_dims_struct or {}
        var_types_struct = var_types_struct or {}

        non_updateable = self._field_names_set.intersection(kwargs)
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
            if dt is ParNotSet:
                dt = None

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

    def _set_metadata(self, info_data, dt=ParNotSet, param_struct=None, required_params=None):

        if dt is not ParNotSet:
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

            if var_name in self._non_setable_var_names:
                if bin_dim or var_type:
                    raise ValueError(
                        f"Cannot manually set {var_dim_name} or {bin_dim_name} - these are fixed by the MLD "
                        f"specification and dimension.")
                elif var_name == 'delta':
                    info_data[bin_dim_name] = bin_dim = info_data[var_dim_name]
                elif var_name == 'z':
                    info_data[bin_dim_name] = bin_dim = 0
                elif var_name == 'v':
                    continue
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
            elif bin_dim is not None:
                if bin_dim > var_dim:
                    raise ValueError((f"Value of '{bin_dim_name}':{bin_dim} must be non-negative value <= "
                                      f"dimension '{var_dim_name}':{var_dim}"))
                info_data[var_type_name] = atleast_2d_col(list('c' * (var_dim - bin_dim) + 'b' * bin_dim))

        if set(modified_var_names).intersection(self._controllable_var_names):
            concat_bin_dim_name = self._var_to_bin_dim_names_map[self._concat_controllable_var_names[0]]
            concat_var_type_name = self._var_to_type_names_map[self._concat_controllable_var_names[0]]

            info_data[concat_bin_dim_name] = sum(
                [info_data[self._var_to_bin_dim_names_map[var_name]] for var_name in self._controllable_var_names]
            )
            info_data[concat_var_type_name] = np.vstack(
                [info_data[self._var_to_type_names_map[var_name]] for var_name in
                 self._controllable_var_names]
            )

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
        def _process_funcs(func, axis, args):
            if isinstance(args, collections.Sequence):
                if isinstance(args[0], str):
                    args = StructDict.get_sub_list(mld_mat_shapes_struct, args)
                    return tuple(func(args))[axis]
                elif np.isscalar(args[0]):
                    return func(args)
                else:
                    return _process_funcs(func,
                                          axis,
                                          [_process_funcs(_func, _axis, _args) for (_func, _axis, _args) in args]
                                          )
            else:
                raise ValueError("Incorrect arguments")

        mld_dims = StructDict()
        for dim_name, (func, axis, args) in self._mld_dim_map.items():
            mld_dims[dim_name] = _process_funcs(func, axis, args)
        return mld_dims

    @staticmethod
    def _get_num_var_bin(var_types_vect):
        if var_types_vect is None:
            return None
        else:
            return (np.asanyarray(var_types_vect, dtype=np.str) == 'b').sum()


@versioned(versioned_sub_objects=('mld_info',))
class MldModel(MldBase):
    __internal_names = ('_mld_info', '_mld_type', '_shapes_struct', '_all_empty_mats', '_all_zero_mats')

    _MLD_MODEL_TYPE_NAMES = ['numeric', 'callable', 'symbolic']
    MldModelTypesNamedTup = NamedTuple('MldModelTypes', _MLD_MODEL_TYPE_NAMES)
    MldModelTypes = MldModelTypesNamedTup._make(_MLD_MODEL_TYPE_NAMES)

    _MLD_MODEL_MAT_TYPES_NAMES = ['state_input', 'output', 'constraint']
    MldModelMatTypesNamedTup = NamedTuple('MldModelMatTypes', _MLD_MODEL_MAT_TYPES_NAMES)
    MldModelMatTypes = MldModelMatTypesNamedTup._make(_MLD_MODEL_MAT_TYPES_NAMES)

    ## Mld Model Config
    _state_input_mat_names = ['A', 'B1', 'B2', 'B3', 'B4', 'b5']
    _state_input_mat_names_private = ['_zeros_G_state_input', '_zeros_Psi_state_input']
    _state_input_mat_names_all = _state_input_mat_names + _state_input_mat_names_private

    _output_mat_names = ['C', 'D1', 'D2', 'D3', 'D4', 'd5']
    _output_mat_names_private = ['_zeros_G_output', '_zeros_Psi_output']
    _output_mat_names_all = _output_mat_names + _output_mat_names_private

    _constraint_mat_names = ['E', 'F1', 'F2', 'F3', 'F4', 'f5', 'G', 'Psi']
    _constraint_mat_names_private = []
    _constraint_mat_names_all = _constraint_mat_names + _constraint_mat_names_private

    _offset_vect_names = MldModelMatTypesNamedTup('b5', 'd5', 'f5')

    _data_types = MldModelMatTypes
    _data_layout = OrderedDict([(MldModelMatTypes.state_input, _state_input_mat_names),
                                (MldModelMatTypes.output, _output_mat_names),
                                (MldModelMatTypes.constraint, _constraint_mat_names)])

    _sys_mat_names = set([data for data_type in _data_layout.values() for data in data_type])
    _sys_mat_names_private = set(_state_input_mat_names_private + _output_mat_names_private)

    _field_names = sorted(_sys_mat_names | _sys_mat_names_private)
    _field_names_set = frozenset(_field_names)

    def __init__(self, system_matrices=None, dt=ParNotSet, param_struct=None, bin_dims_struct=None,
                 var_types_struct=None,
                 **kwargs):
        super(MldModel, self).__init__()
        super(MldModel, self).update(dict.fromkeys(self._field_names_set, np.empty(shape=(0, 0))))
        self._mld_info = MldInfo()
        self._mld_type = None
        self._shapes_struct = StructDict.fromkeys(self._field_names_set, (0, 0))
        self._all_empty_mats = set()
        self._all_zero_mats = set()

        self.update(system_matrices=system_matrices, dt=dt, param_struct=param_struct, bin_dims_struct=bin_dims_struct,
                    var_types_struct=var_types_struct, from_init=True, **kwargs)

    @property
    def mld_info(self):
        return self._mld_info

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
                         "y(k)   = C*x(k) + D1*u(k) + D2*delta(k) + D3*z(k) + D4*omega(k) + d5\n"
                         "E*x(k) + F1*u(k) + F2*delta(k) + F3*z(k) + F4*omega(k) + G*y(k) + Psi*mu(k) <= f5\n"
                         "                                                                      mu(k) >= 0\n"
                         f"mld_type : {self.mld_type}\n"
                         "\n"
                         "with:\n")
        mld_model_repr = base_repr.replace('{', '{\n' + mld_model_str, 1)
        return mld_model_repr

    @_process_mld_args_decor
    @increments_version_decor
    def update(self, system_matrices=None, dt=ParNotSet, param_struct=None, bin_dims_struct=None, var_types_struct=None,
               **kwargs):
        from_init = kwargs.pop('from_init', False)
        mld_info_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in MldInfo._field_names_set}

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
            except TypeError as te:
                raise TypeError(f"Argument:'system_matrices' must be dictionary like: {te.args[0]}")

        new_sys_mats = self.as_base_dict()
        new_shapes_struct = self._shapes_struct.as_base_dict()
        try:
            for sys_matrix_id, system_matrix in creation_matrices.items():
                if sys_matrix_id in self._sys_mat_names:
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
                        new_shapes_struct[sys_matrix_id] = get_expr_shape(system_matrix)
                elif sys_matrix_id not in self._field_names_set:
                    if sys_matrix_id in kwargs:
                        raise ValueError("Invalid matrix name in kwargs: {}".format(sys_matrix_id))
                    else:
                        raise ValueError("Invalid matrix name in system_matrices: {}".format(sys_matrix_id))
        except AttributeError as ae:
            raise TypeError(f"Argument:'system_matrices' must be dictionary like: {ae.args[0]}")

        if from_init and creation_matrices.get('C') is None:
            C = np.eye(*(get_expr_shape(new_sys_mats['A'])), dtype=np.int)
            C.setflags(write=False)
            creation_matrices['C'] = C
            new_sys_mats['C'] = C
            new_shapes_struct['C'] = C.shape

        if new_shapes_struct != self._shapes_struct:
            mld_dims = self._mld_info._get_mld_dims(new_shapes_struct)
            new_sys_mats, new_shapes_struct = (
                self._validate_sys_matrix_shapes(new_sys_mats, new_shapes_struct, mld_dims))
            self._set_all_zero_or_empty(new_sys_mats, new_sys_mats.keys())
        else:
            self._set_all_zero_or_empty(new_sys_mats, creation_matrices.keys())
            mld_dims = None

        self._shapes_struct.update(new_shapes_struct)
        new_sys_mats = self._set_mld_type(new_sys_mats)
        super(MldModel, self).update(new_sys_mats)

        if self.mld_type in (self.MldModelTypes.callable, self.MldModelTypes.symbolic):
            required_params = self._get_required_params()
        else:
            required_params = None

        self.mld_info.update(mld_info_data=mld_dims, dt=dt, param_struct=param_struct, bin_dims_struct=bin_dims_struct,
                             var_types_struct=var_types_struct, required_params=required_params, **mld_info_kwargs)

    # TODO not finished - needs to include output constraints, interpolation and datetime handling
    LSimStruct = struct_prop_dict('LSimStruct', ['t_out', 'y_out', 'x_out', 'con_out', 'x_out_k1'], sorted_repr=False)

    def lsim(self, x_k=None, u=None, delta=None, z=None, mu=None, v=None, omega=None, t=None, dt=ParNotSet):
        dt = dt if dt is not ParNotSet else self.mld_info.dt
        m_dims = self.mld_info.var_dims_struct
        inputs_struct = StructDict()

        if v is not None:
            if not is_all_None(u, delta, z, mu):
                raise ValueError(
                    "Either supply concatenated input in 'v' or supply individual inputs "
                    "'u', 'delta', 'z' and 'mu', but not both.")
            else:
                inputs_struct.v = v

        else:
            inputs_struct.update(
                u=atleast_2d_col(u) if u is not None else atleast_2d_col([]),
                delta=atleast_2d_col(delta) if delta is not None else atleast_2d_col([]),
                z=atleast_2d_col(z) if z is not None else atleast_2d_col([]),
                mu=atleast_2d_col(mu) if mu is not None else atleast_2d_col([]),
            )

        inputs_struct.omega = atleast_2d_col(omega) if omega is not None else atleast_2d_col([])

        max_input_samples = get_expr_shapes(inputs_struct, get_max_dim=True)[0]
        if t is None:
            out_samples = max_input_samples
            stoptime = (out_samples - 1) * dt
        else:
            stoptime = t[-1]
            out_samples = int(np.floor(stoptime / dt)) + 1

        for input_name, input in inputs_struct.items():
            input_dim = m_dims[input_name]
            if input.shape[1] == input_dim:
                if input_dim == 0 and input.shape[0] != out_samples:  # ensure null inputs have correct dimension
                    inputs_struct[input_name] = inputs_struct[input_name].reshape(out_samples, 0)
                elif t is None and input.shape[0] != max_input_samples:
                    raise ValueError(
                        f"Invalid shape for input sequence'{input_name}':{input.shape}, row dimension must be equal "
                        f"to maximum row dimension of all input sequences, i.e. '({max_input_samples},*)')"
                    )
                elif t is not None and input.shape[0] != t.size:
                    raise ValueError(
                        f"Invalid shape for input sequence'{input_name}':{input.shape}, row dimension must be equal "
                        f"to size of t, i.e. '({t.size},*)')"
                    )
            else:
                raise ValueError(
                    f"Invalid shape for input sequence '{input_name}':{input.shape}, column dimension must be equal "
                    f"to mld model dim '{self.mld_info._var_to_dim_names_map[input_name]}', i.e. '(*,{input_dim})')"
                )

        v = inputs_struct.get('v')
        if v is not None:
            u, delta, z, mu = np.split(v, [m_dims.u,
                                           m_dims.u + m_dims.delta,
                                           m_dims.u + m_dims.delta + m_dims.z], axis=1)
        else:
            u = inputs_struct.u
            delta = inputs_struct.delta
            z = inputs_struct.z
            mu = inputs_struct.mu

        omega = inputs_struct.omega

        # Pre-build output arrays
        x_out_k1 = np.zeros((out_samples + 1, m_dims.x))
        y_out = np.zeros((out_samples, m_dims.y))
        con_out = np.zeros((out_samples, self.mld_info.n_constraints), dtype=np.bool)
        t_out = atleast_2d_col(np.linspace(0.0, stoptime, num=out_samples))

        # Check initial condition
        if x_k is None:
            x_out_k1[0, :] = np.zeros((m_dims.x,))
        else:
            x_out_k1[0, :] = np.asanyarray(x_k).reshape(1, m_dims.x)

        # Pre-interpolate inputs into the desired time steps
        # todo interpolate inputs

        # return x_out_k1, u_dt, delta, z, omega
        # Simulate the system
        for k in range(0, out_samples):
            x_out_k1[k + 1, :] = (
                    self.A @ x_out_k1[k, :] + self.B1 @ u[k, :] + self.B2 @ delta[k, :]
                    + self.B3 @ z[k, :] + self.B4 @ omega[k, :] + self.b5.T)
            y_out[k, :] = (
                    self.C @ x_out_k1[k, :] + self.D1 @ u[k, :] + self.D2 @ delta[k, :]
                    + self.D3 @ z[k, :] + self.D4 @ omega[k, :] + self.d5.T)
            con_out[k, :] = (
                    (self.E @ x_out_k1[k, :] + self.F1 @ u[k, :] + self.F2 @ delta[k, :]
                     + self.F3 @ z[k, :] + self.F4 @ omega[k, :] + self.G @ y_out[k, :] +
                     self.Psi @ mu[k, :]) <= self.f5.T)

        x_out = x_out_k1[:-1, :]  # remove last point for equal length output arrays.

        return self.LSimStruct(t_out=t_out, y_out=y_out, x_out=x_out, con_out=con_out, x_out_k1=x_out_k1)

    LSimStruct_k = struct_prop_dict('LSimStruct_k', ['x_k1']+list(MldInfo._var_names)+ ['con'],
                                    sorted_repr=False)

    def lsim_k(self, x_k=None, u_k=None, delta_k=None, z_k=None, mu_k=None, v_k=None, omega_k=None,
               solver=None, cons_tol=1e-6) -> LSimStruct_k:
        m_dims = self.mld_info.var_dims_struct

        if x_k is None:
            x_k = np.zeros((m_dims.x, 1))
        else:
            x_k = np.asanyarray(x_k).reshape(m_dims.x, 1)

        omega_k = atleast_2d_col(omega_k) if omega_k is not None else np.empty((0, 1))

        if v_k is not None:
            if not is_all_None(u_k, delta_k, z_k, mu_k):
                raise ValueError(
                    "Either supply concatenated input in 'v_k' or supply individual inputs "
                    "'u_k', 'delta_k', 'z_k' and 'mu_k', but not both.")
            else:
                v_k = atleast_2d_col(v_k)
                u_k = v_k[:m_dims.u]
                delta_k = v_k[m_dims.u:m_dims.u + m_dims.delta]
                z_k = v_k[m_dims.u + m_dims.delta:m_dims.u + m_dims.delta + m_dims.z]
                mu_k = v_k[m_dims.u + m_dims.delta + m_dims.z:]
        else:
            u_k = atleast_2d_col(u_k) if u_k is not None else np.empty((0, 1))
            mu_k = atleast_2d_col(mu_k) if mu_k is not None else np.zeros((m_dims.mu, 1))
            if is_any_None(delta_k, z_k):
                delta_k, z_k = (
                    self._compute_aux(x_k=x_k, u_k=u_k, delta_k=delta_k, z_k=z_k, mu_k=mu_k, omega_k=omega_k,
                                      m_dims=m_dims, solver=solver))
            else:
                delta_k = atleast_2d_col(delta_k)
                z_k = atleast_2d_col(z_k)

            v_k = np.vstack((u_k, delta_k, z_k, mu_k))

        x_k1 = (self.A @ x_k + self.B1 @ u_k + self.B2 @ delta_k + self.B3 @ z_k + self.B4 @ omega_k + self.b5)
        y_k = (self.C @ x_k + self.D1 @ u_k + self.D2 @ delta_k + self.D3 @ z_k + self.D4 @ omega_k + self.d5)
        con_k = (self.E @ x_k + self.F1 @ u_k + self.F2 @ delta_k
                 + self.F3 @ z_k + self.F4 @ omega_k + self.G @ y_k +
                 self.Psi @ mu_k - self.f5 <= cons_tol)

        return self.LSimStruct_k(x_k1=x_k1, x=x_k,
                                 u=u_k, delta=delta_k, z=z_k, mu=mu_k, v=v_k,
                                 y=y_k, omega=omega_k,
                                 con=con_k)

    def _compute_aux(self, x_k=None, u_k=None, delta_k=None, z_k=None, mu_k=None, omega_k=None, m_dims=None,
                     solver=None):

        if delta_k is None:
            if m_dims.delta:
                delta_k = cvx.Variable((m_dims.delta, 1), boolean=True)
            else:
                delta_k = np.empty((0, 1))
        else:
            delta_k = atleast_2d_col(delta_k)

        if z_k is None:
            if m_dims.z:
                z_k = cvx.Variable((m_dims.z, 1))
            else:
                z_k = np.empty((0, 1))
        else:
            z_k = atleast_2d_col(z_k)

        if delta_k.size or z_k.size:
            if m_dims.y:
                y_k = cvx.Variable((m_dims.y, 1))
            else:
                y_k = np.empty((0, 1))

            cons = []
            if y_k.size:
                cons.append(
                    y_k == (matmul(self.C, x_k) +
                            matmul(self.D1, u_k) +
                            matmul(self.D2, delta_k) +
                            matmul(self.D3, z_k) +
                            matmul(self.D4, omega_k) +
                            self.d5))

            cons.append(
                matmul(self.E, x_k) +
                matmul(self.F1, u_k) +
                matmul(self.F2, delta_k) +
                matmul(self.F3, z_k) +
                matmul(self.F4, omega_k) +
                matmul(self.G, y_k) +
                matmul(self.Psi, mu_k) <= self.f5
            )

            prob = cvx.Problem(cvx.Minimize(cvx.Constant(0)), cons)
            prob.solve(verbose=False, solver=solver or cvx.GUROBI)

            # if prob.status not in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
            #     raise ValueError(f"mld_model is infeasible")

            if isinstance(delta_k, cvx_e.Expression) and delta_k.size:
                delta_k = np.abs(delta_k.value) if delta_k.value is not None else atleast_2d_col(np.NaN)
            if isinstance(z_k, cvx_e.Expression) and z_k.size:
                z_k = z_k.value if z_k.value is not None else atleast_2d_col(np.NaN)

        return delta_k, z_k

    def to_numeric(self, param_struct=None, dt=ParNotSet, copy=False):
        param_struct = param_struct if param_struct is not None else self.mld_info.param_struct

        if dt is ParNotSet:
            if param_struct and param_struct.get('dt', ParNotSet) is not ParNotSet:
                dt = param_struct['dt']
            else:
                dt = self.mld_info.dt

        if param_struct is not None:
            param_struct = _copy(param_struct)
            if param_struct.get('dt', ParNotSet) is not ParNotSet:
                param_struct['dt'] = dt

        if self.mld_info.dt is None and dt is not None:
            raise NotImplementedError("Discretization required")
            # todo discretisation required

        if self.mld_type == self.MldModelTypes.callable:
            dict_numeric = {mat_id: mat_callable(param_struct=param_struct) for mat_id, mat_callable in
                            self.items()}
        elif self.mld_type == self.MldModelTypes.symbolic:
            print("Performance warning, mld_type is not callable had to convert to callable")
            callable_mld = self.to_callable(param_struct=param_struct, dt=dt)
            return callable_mld.to_numeric(param_struct=param_struct, copy=copy)
        else:
            dict_numeric = _deepcopy(self.as_base_dict()) if copy else self.as_base_dict()

        # make all system matrices read_only
        for mat in dict_numeric.values():
            mat.setflags(write=False)

        mld_type = self.MldModelTypes.numeric
        numeric_mld = self._constructor_from_self(items=dict_numeric, copy_instance_attr=True,
                                                  deepcopy_instance_attr=copy,
                                                  instance_attr_override={'_mld_type': mld_type})
        numeric_mld.mld_info._base_dict_update(param_struct=param_struct, dt=dt)
        return numeric_mld

    def to_callable(self, param_struct=None, dt=ParNotSet, copy=False):
        param_struct = param_struct if param_struct is not None else self.mld_info.param_struct

        if dt is ParNotSet:
            if param_struct and param_struct.get('dt', ParNotSet) is not ParNotSet:
                dt = param_struct['dt']
            else:
                dt = self.mld_info.dt

        if param_struct is not None:
            param_struct = _copy(param_struct)
            if param_struct.get('dt', ParNotSet) is not ParNotSet:
                param_struct['dt'] = dt

        if self.mld_type in (self.MldModelTypes.numeric, self.MldModelTypes.symbolic):
            dict_callable = {}
            for sys_matrix_id, system_matrix in self.items():
                dict_callable[sys_matrix_id] = CallableMatrix(system_matrix, sys_matrix_id)
        else:
            dict_callable = _deepcopy(self.as_base_dict()) if copy else self.as_base_dict()

        mld_type = self.MldModelTypes.callable
        callable_mld = self._constructor_from_self(items=dict_callable, copy_instance_attr=True,
                                                   deepcopy_instance_attr=copy,
                                                   instance_attr_override={'_mld_type': mld_type})
        callable_mld.mld_info._base_dict_update(param_struct=param_struct, dt=dt)
        return callable_mld

    def _set_mld_type(self, mld_data):
        is_symbolic = (isinstance(sys_mat, (sp.Expr, sp.Matrix)) for sys_mat_id, sys_mat in
                       mld_data.items())
        if any(is_symbolic):
            self._mld_type = self.MldModelTypes.symbolic
        elif any(callable(sys_mat) for sys_mat_id, sys_mat in mld_data.items()):
            for sys_mat_id, sys_mat in mld_data.items():
                if not callable(sys_mat):
                    mld_data[sys_mat_id] = CallableMatrix(sys_mat, sys_mat_id)
            self._mld_type = self.MldModelTypes.callable
        else:
            self._mld_type = self.MldModelTypes.numeric

        return mld_data

    def _set_all_zero_or_empty(self, mld_data, mat_ids=None):
        if mat_ids is not None:
            changed_mats = StructDict.sub_struct_fromdict(mld_data, mat_ids)
        else:
            changed_mats = mld_data
        for mat_id, mat in changed_mats.items():
            if isinstance(mat, CallableMatrix):
                is_empty = mat.is_empty
                is_all_zero = mat.is_all_zero
            elif isinstance(mat, sp.Matrix):
                is_empty = False if np.prod(mat.shape) else True
                is_all_zero = mat.is_zero
            else:
                is_empty = False if mat.size else True
                is_all_zero = np.all(mat == 0)

            self._all_empty_mats.add(mat_id) if is_empty else self._all_empty_mats.discard(mat_id)
            self._all_zero_mats.add(mat_id) if is_all_zero else self._all_zero_mats.discard(mat_id)

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

            # var is a constant vector in state_input or output equations
            if var_dim is None and sys_dim and mat_type is not self.MldModelMatTypes.constraint:
                var_dim = 1

            if var_dim and var_dim > 0:
                new_shape = (sys_dim, var_dim)
                zero_empty_mat = np.zeros(new_shape, dtype=np.int)
            else:
                new_shape = (sys_dim, 0)
                zero_empty_mat = np.empty(shape=new_shape, dtype=np.int)

            if callable(mld_data[mat_id]):
                zero_empty_mat = CallableMatrix(zero_empty_mat, mat_id)

            return zero_empty_mat, new_shape

        _var_const_dim_names = self._mld_info._var_and_const_to_dim_names_map.values()

        sys_dim_name_triad = self.MldModelMatTypesNamedTup._make(self._mld_info._sys_dim_names)
        sys_mat_ids_zip = zip(self._state_input_mat_names_all, self._output_mat_names_all,
                              self._constraint_mat_names_all)
        for var_dim_name, sys_mat_name_triad in zip(_var_const_dim_names, sys_mat_ids_zip):
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
        concat_sys_mats = StructDict.fromkeys(MldModel._field_names_set, [])
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


@versioned(versioned_sub_objects=('mld_numeric', 'mld_callable', 'mld_symbolic'))
class MldSystemModel(VersionMixin):
    MldNames = MldModel.MldModelTypesNamedTup(numeric='mld_numeric', callable='mld_callable', symbolic='mld_symbolic')

    def __init__(self, mld_numeric=None, mld_callable=None, mld_symbolic=None, param_struct=None, copy=False):
        super(MldSystemModel, self).__init__()
        self._param_struct = None
        self._mld_numeric = None
        self._mld_callable = None
        self._mld_symbolic = None
        self.update_mld(mld_numeric=mld_numeric, mld_callable=mld_callable, mld_symbolic=mld_symbolic,
                        param_struct=param_struct, copy=copy, missing_param_check=True)

    @increments_version_decor
    def update_mld(self, mld_numeric=None, mld_callable=None, mld_symbolic=None, param_struct=None,
                   param_struct_subset=None, copy=False, missing_param_check=True, invalid_param_check=False, **kwargs):

        mlds = MldModel.MldModelTypesNamedTup(numeric=mld_numeric, callable=mld_callable, symbolic=mld_symbolic)
        if num_not_None(*mlds) > 1:
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
    def mld_numeric(self) -> MldModel:
        return self._mld_numeric

    @property
    def mld_callable(self) -> MldModel:
        return self._mld_callable

    @property
    def mld_symbolic(self) -> MldModel:
        return self._mld_symbolic

    @property
    def param_struct(self):
        return self._param_struct

    @param_struct.setter
    def param_struct(self, param_struct):
        self.update_param_struct(param_struct=param_struct)

    @increments_version_decor
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
                        invalid_param_check=True, copy=False, **kwargs) -> MldModel:

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
            if copy:
                return self._mld_numeric.deepcopy()
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


class PvMldSystemModel(MldSystemModel):

    def _process_param_struct_args(self, f_kwargs=None,
                                   param_struct=ParNotSet, param_struct_subset=None):

        _param_struct = self._validate_param_struct(
            param_struct=f_kwargs.get('param_struct'), param_struct_subset=param_struct_subset,
            missing_param_check=False, invalid_param_check=False)

        if param_struct is None:
            f_kwargs['param_struct'] = _param_struct

        return f_kwargs

    @process_method_args_decor(_process_param_struct_args)
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

    @process_method_args_decor(_process_param_struct_args)
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


if __name__ == '__main__':
    a = dict.fromkeys(MldModel._field_names_set, np.ones((2, 2)))
    a.update(b5=np.ones((2, 1)), d5=np.ones((2, 1)), f5=np.ones((2, 1)))
    mld = MldModel(a)
    mld2 = MldModel({'A': 1})
    #
    # from models.model_generators import DewhModelGenerator
    # b = DewhModelGenerator().gen_dewh_symbolic_sys_matrices()
    # c = b.to_callable()
