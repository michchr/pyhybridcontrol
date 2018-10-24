import numpy as np
import sympy as sp
import scipy.sparse as scs
import scipy.linalg as scl
import functools
from enum import Enum
from copy import deepcopy as _deepcopy

import inspect

from sortedcontainers import SortedDict
from utils.structdict import StructDict, SortedStructDict


class MldBase(SortedStructDict):
    _internal_names = []
    _internal_names_set = StructDict._internal_names_set.union(_internal_names)
    _allowed_data_set = set()

    def _process_args(self, args, kwonlyargs_names=None):
        if args and (args[0] is None or callable(args[0])):
            _key = args[0]
            args = args[1:]
        else:
            _key = None

        if kwonlyargs_names is not None:
            named_arg_dict = {kwonlyargs_names[i]: arg for (i, arg) in enumerate(args)}
            named_arg_struct = StructDict(named_arg_dict)
        else:
            named_arg_struct = StructDict()

        return _key, args, named_arg_struct

    def _init_std_attributes(self):
        _sdict = super(MldBase, self)
        _sdict._init_std_attributes()
        self._sdict_init = _sdict.__init__
        self._sdict_setitem = _sdict.__setitem__
        self._sdict_update = _sdict.update

    # noinspection PyMissingConstructor
    def __init__(self, *args, **kwargs):
        _key, _, _ = self._process_args(args)
        if _key and isinstance(self, SortedDict):
            # noinspection PyUnresolvedReferences
            self._sdict_init(_key)
        self._sdict_update(dict.fromkeys(self._allowed_data_set))

        for attr in self._internal_names_set:
            self.__setattr__(attr, None)

    def __setitem__(self, key, value):
        if key in self.keys():
            self.update(**{key: value})
        elif key in self._internal_names_set:
            object.__setattr__(self, key, value)
        else:
            raise KeyError("key:'{}' is not valid or does not exist".format(key))

    def __copy__(self):
        return self.__class__(self)

    def __reduce__(self):
        return (self.__class__, (self._key, list(self.items())))


_mld_dim_map = {
    'nx': ('A', 'E1'),
    'nu': ('B1', 'E2'),
    'ndelta': ('B2', 'E3'),
    'nz': ('B3', 'E4'),
    'nomega': ('B4', 'E5'),
    'n_cons': ('d',)
}

class MldVarInfo(MldBase):
    _internal_names = ['_mld_model']
    _internal_names_set = MldBase._internal_names_set.union(_internal_names)

    _valid_var_types = ['c', 'b']
    _state_names_user_setable = ['x', 'u', 'omega']
    _state_names_non_setable = ['delta', 'z']

    _state_names = _state_names_user_setable + _state_names_non_setable
    _state_dim_names = ["".join(['n', name]) for name in _state_names]
    _con_dim_name = ['n_cons']
    _var_type_names = ["".join(['var_type_', name]) for name in _state_names]
    _bin_dim_names = ["".join([dim_name, '_l']) for dim_name in _state_dim_names]
    _bin_dim_names_non_setable = ["".join([dim_name, '_l']) for dim_name in _state_names_non_setable]

    _allowed_data_set = set(_state_dim_names + _con_dim_name + _var_type_names + _bin_dim_names)

    def __init__(self, *args, mld_model=None, bin_dims_struct=None, var_types_struct=None, **kwargs):
        super(MldVarInfo, self).__init__(*args, **kwargs)

        kwonlyargs_names = inspect.getfullargspec(self.__init__).kwonlyargs
        _, args, named_arg_struct = self._process_args(args, kwonlyargs_names)

        mld_model = mld_model or named_arg_struct.get('mld_model')
        bin_dims_struct = bin_dims_struct or named_arg_struct.get('bin_dims_struct')
        var_types_struct = var_types_struct or named_arg_struct.get('var_types_struct')

        self.update(*args, mld_model=mld_model, bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct,
                    **kwargs)

    def __reduce__(self):
        mld_data = list(self.mld_model.items()) if isinstance(self.mld_model, MldModel) else None
        return (self.__class__, (self._key, mld_data, None, self.var_types_struct))

    @property
    def var_types_struct(self):
        return StructDict(
            {var_type_name: self.get(var_type_name) for var_type_name in self._var_type_names})

    @property
    def mld_model(self):
        return self._mld_model

    @mld_model.setter
    def mld_model(self, value):
        raise AttributeError("Cannot set mld_model directly use MldVarInfo.update() instead.")

    def update(self, *args, mld_model=None, bin_dims_struct=None, var_types_struct=None, **kwargs):

        kwonlyargs_names = inspect.getfullargspec(self.update).kwonlyargs
        _, args, named_arg_struct = self._process_args(args, kwonlyargs_names)

        mld_model = mld_model or named_arg_struct.get('mld_model')
        bin_dims_struct = bin_dims_struct or named_arg_struct.get('bin_dims_struct')
        var_types_struct = var_types_struct or named_arg_struct.get('var_types_struct')

        if kwargs:
            bin_dims_struct, var_types_struct = self._update_set_struct_from_kwargs(
                bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct, kwargs=kwargs)


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
            self._set_var_info(bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct)

    def _set_var_info(self, bin_dims_struct=None, var_types_struct=None):
        # either set to new value, old value, or zero in that order - never None
        bin_dims_struct = bin_dims_struct or {}
        var_types_struct = var_types_struct or {}

        _temp_data = StructDict(self.items())
        zip_gen = zip(self._state_dim_names, self._bin_dim_names, self._var_type_names)

        for (state_dim_name, bin_dim_name, var_type_name) in zip_gen:
            bin_dim = bin_dims_struct.get(bin_dim_name)
            var_type = var_types_struct.get(var_type_name)
            state_dim = self.get(state_dim_name)

            if bin_dim_name in self._bin_dim_names_non_setable:
                if bin_dim:
                    raise ValueError(
                        "Cannot manually set ndelta_l, nz_l - these are fixed by the MLD specification and dimension.")
                elif state_dim_name == 'ndelta':
                    bin_dim = self[state_dim_name]
                    _temp_data[bin_dim_name] = bin_dim
                elif state_dim_name == 'nz':
                    bin_dim = 0
                    _temp_data[bin_dim_name] = bin_dim
            else:
                bin_dim_queue = [self._get_num_var_bin(var_type), bin_dim, self.get(bin_dim_name)]
                bin_dim = next((item for item in bin_dim_queue if item is not None), 0)
                _temp_data[bin_dim_name] = bin_dim

            if var_type is not None:
                var_type = self._check_var_types_vect_valid(var_type, _temp_data, bin_dim_name)
                if var_type.size != self[state_dim_name]:
                    raise ValueError(
                        "Dimension of '{0}' must match dimension: '{1}'".format(var_type_name, state_dim_name))
                _temp_data[var_type_name] = var_type
            else:
                try:
                    _temp_data[var_type_name] = np.hstack(
                        [np.repeat('c', state_dim - bin_dim), np.repeat('b', bin_dim)])
                except ValueError:
                    raise ValueError(
                        "Value of '{0}':{1} must be non-negative value <= dimension '{2}':{3}".format(
                            bin_dim_name, bin_dim, state_dim_name, self[state_dim_name]))

        self._sdict_update(_temp_data)

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
        for state_dim_name, sys_matrix_ids in _mld_dim_map.items():
            system_matrices = (self.mld_model.get(sys_id) for sys_id in sys_matrix_ids)
            if state_dim_name != 'n_cons':
                self._sdict_setitem(state_dim_name, _get_expr_shapes(*system_matrices, get_max_dim=True)[1])
            else:
                self._sdict_setitem(state_dim_name, _get_expr_shapes(*system_matrices, get_max_dim=True)[0])

    def _update_set_struct_from_kwargs(self, bin_dims_struct=None, var_types_struct=None, kwargs=None):
        bin_dims_struct = bin_dims_struct or StructDict()
        var_types_struct = var_types_struct or StructDict()

        if kwargs:
            for key in self._bin_dim_names:
                if key in kwargs:
                    bin_dims_struct[key] = kwargs.pop(key)
            for key in self._var_type_names:
                if key in kwargs:
                    var_types_struct[key] = kwargs.pop(key)

        return bin_dims_struct, var_types_struct

    @staticmethod
    def _get_num_var_bin(var_types_vect):
        if var_types_vect is None:
            return None
        else:
            var_types_vect_flat = np.ravel(var_types_vect)
            return (var_types_vect_flat == 'b').sum()


_mld_types = ['numeric', 'symbolic', 'callable']
MldType = Enum('MldType', _mld_types)


class MldModel(MldBase):
    _internal_names = ['mld_info', 'mld_type']
    _internal_names_set = MldBase._internal_names_set.union(_internal_names)

    _state_matrix_names = ['A', 'B1', 'B2', 'B3', 'B4', 'b5']
    _con_matrix_names = ['E1', 'E2', 'E3', 'E4', 'E5', 'd']

    _allowed_data_set = set(_state_matrix_names + _con_matrix_names)

    def __init__(self, *args, system_matrices=None, bin_dims_struct=None, var_types_struct=None, **kwargs):
        super(MldModel, self).__init__(*args, **kwargs)
        self.update(dict.fromkeys(self._allowed_data_set, np.empty(shape=(0, 0))))

        kwonlyargs_names = inspect.getfullargspec(self.__init__).kwonlyargs
        _, args, named_arg_struct = self._process_args(args, kwonlyargs_names)

        system_matrices = system_matrices or named_arg_struct.get('system_matrices')
        bin_dims_struct = bin_dims_struct or named_arg_struct.get('bin_dims_struct')
        var_types_struct = var_types_struct or named_arg_struct.get('var_types_struct')

        self.mld_type = None

        self.update(*args, system_matrices=system_matrices, bin_dims_struct=bin_dims_struct,
                    var_types_struct=var_types_struct, **kwargs)

    def __reduce__(self):
        return (self.__class__, (self._key, list(self.items()), None, self.mld_info.var_types_struct))

    def update(self, *args, system_matrices=None, bin_dims_struct=None, var_types_struct=None, **kwargs):
        kwonlyargs_names = inspect.getfullargspec(self.update).kwonlyargs
        _, args, named_arg_struct = self._process_args(args, kwonlyargs_names)

        system_matrices = system_matrices or named_arg_struct.get('system_matrices')
        bin_dims_struct = bin_dims_struct or named_arg_struct.get('bin_dims_struct')
        var_types_struct = var_types_struct or named_arg_struct.get('var_types_struct')

        if args and isinstance(args[0], self.__class__):
            self._sdict_update(args[0])
            self.mld_info = args[0].mld_info
            system_matrices = None

        if not isinstance(self.mld_info, MldVarInfo):
            self.mld_info = MldVarInfo()

        if kwargs:
            bin_dims_struct, var_types_struct = self.mld_info._update_set_struct_from_kwargs(
                bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct, kwargs=kwargs)

        if system_matrices and kwargs:
            raise ValueError("Individual matrix arguments cannot be set if 'system_matrices' argument is set")
        creation_matrices = system_matrices or kwargs

        if not isinstance(creation_matrices, dict):
            try:
                creation_matrices = dict(creation_matrices)
            except Exception:
                pass

        try:
            _temp_data = StructDict(self.items())
            for sys_matrix_id, system_matrix in creation_matrices.items():
                if sys_matrix_id in self._allowed_data_set:
                    old_val = self.get(sys_matrix_id)
                    if isinstance(system_matrix, (sp.Expr)):
                        system_matrix = sp.Matrix([system_matrix])
                    elif not isinstance(system_matrix, sp.Matrix) and not callable(system_matrix):
                        system_matrix = np.atleast_2d(system_matrix)
                    _temp_data[sys_matrix_id] = system_matrix if system_matrix is not None else old_val
                else:
                    if kwargs:
                        raise ValueError("Invalid matrix name in kwargs: {}".format(sys_matrix_id))
                    else:
                        raise ValueError("Invalid matrix name in system_matrices: {}".format(sys_matrix_id))
        except AttributeError:
            raise TypeError("Argument:'system_matrices' must be dictionary like")

        try:
            self.verify_shapes_valid(_temp_data)
            self._set_mld_type(_temp_data)
            super(MldModel, self).update(_temp_data)
        except ValueError as ve:
            raise ve

        self.mld_info.update(self, bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct)

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

    def verify_shapes_valid(self, mld_data=None):
        if mld_data is None:
            mld_data = self

        shapes_struct = _get_expr_shapes(mld_data)

        A_shape = shapes_struct.A
        d_shape = shapes_struct.d

        if A_shape[0] != A_shape[1] and 0 not in A_shape:
            raise ValueError("Invalid shape for matrix A:'{}', must be a square matrix or scalar".format(A_shape))

        for state_matrix_id, con_matrix_id in zip(self._state_matrix_names, self._con_matrix_names):
            state_matrix_shape = shapes_struct[state_matrix_id]
            con_matrix_shape = shapes_struct[con_matrix_id]
            if state_matrix_shape[0] not in (A_shape[0], 0):
                raise ValueError(
                    "Invalid shape for state matrix/vector:'{0}':{1}, must have same row dimension as state_matrix "
                    "'A', i.e. '({2},*)')".format(state_matrix_id, state_matrix_shape, A_shape[0]))

            if (con_matrix_shape[1] not in (state_matrix_shape[1], 0)) and (0 not in state_matrix_shape) and (
                    con_matrix_id != 'd'):
                raise ValueError(
                    "Invalid shape for constraint matrix:'{0}':{1}, must have same column dimension as state matrix "
                    "'{2}', i.e. '(*,{3})')".format(con_matrix_id, con_matrix_shape, state_matrix_id,
                                                    state_matrix_shape[1]))

            if con_matrix_shape[0] not in (d_shape[0], 0):
                raise ValueError(
                    "Invalid shape for constraint matrix:'{0}':{1}, must have same row dimension as constraint "
                    "vector 'd', i.e. '({2},*)')".format(con_matrix_id, con_matrix_shape, d_shape[0]))

        for vect_id in ('b5', 'd'):
            shape_vect = shapes_struct[vect_id]
            if not np.isin(shape_vect, (0, 1)).any():
                raise ValueError(
                    "'{0}' must be of type vector, scalar or null array, currently has shape:{1}".format(vect_id,
                                                                                                         shape_vect))
        return shapes_struct

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

        concat_var_type_info = StructDict.fromkeys(MldVarInfo._var_type_names)
        for var_type_name in concat_var_type_info:
            concat_info_list = []
            for model in mld_model_list:
                concat_info_list.append(model.mld_info[var_type_name])

            concat_var_type_info[var_type_name] = np.hstack(concat_info_list)

        concat_var_type_info.var_type_delta = None
        concat_var_type_info.var_type_z = None

        concat_mld = MldModel(concat_sys_mats, var_types_struct=concat_var_type_info)

        return concat_mld


def _get_expr_shape(expr):
    if expr is None:
        return (0,0)
    elif np.isscalar(expr) or isinstance(expr, sp.Expr):
        return (1, 1)
    else:
        try:
            expr_shape = expr.shape
            if 0 in expr_shape:
                return (0, 0)
            if len(expr_shape) == 1:
                return (expr_shape[0], 0)
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
                args = [1]*len(inspect.getfullargspec(expr).args)
                return _get_expr_shape(expr(*args))
            except Exception as e:
                raise e
    else:
        raise TypeError("Invalid expression type: '{0}', for expr: '{1!s}'".format(type(expr), expr))


def _get_expr_shapes(*args, get_max_dim=False):
    if not args:
        return None

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, dict):
            shape_struct = StructDict()
            for expr_id, expr in arg.items():
                shape_struct[expr_id] = _get_expr_shape(expr)
            return shape_struct
        else:
            return _get_expr_shape(arg)
    else:
        shapes = []
        for arg in args:
            shapes.append(_get_expr_shape(arg))
        if get_max_dim:
            return tuple(np.maximum(*shapes))
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

#todo functions should be able to take dictionary or positional arguments
def _lambda_wrapper(func, local_syms_str, wrapped_name=None):

    if wrapped_name:
        func.__qualname__ = wrapped_name
        func.__name__ = wrapped_name

    @functools.wraps(func)
    def wrapped(*args, param_struct=None, **kwargs):
        arg_list = []
        if kwargs.pop('empty_call', False):
            arg_list = [1]*len(local_syms_str)
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
    a.update(d=np.ones((2, 1)), b5=np.ones((2, 1)))
    mld = MldModel(a)
    mld2 = MldModel({'A': 1})
    #
    # from models.model_generators import DewhModelGenerator
    # b = DewhModelGenerator().gen_dewh_symbolic_mld_sys_matrices()
    # c = b.to_eval()
