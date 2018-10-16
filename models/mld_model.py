from utils.structdict import SortedStructDict, StructDict

import pprint
import numpy as np
import sympy as sp
import scipy.sparse as scs
import scipy.linalg as scl

# def append_named_call_args(func):
#     def wrapper(self, *args, **kwargs):
#         named_args = inspect.getfullargspec(func).args
#         return func(self, *args, **kwargs, named_call_args=named_args)
#
#     wrapper.__signature__ = inspect.signature(func)
#     return wrapper

def get_expr_shape(expr):
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

    if expr is None or np.isscalar(expr) or isinstance(expr, sp.Expr):
        return (1, 1)
    else:
        raise TypeError("Invalid expression type: '{0}', for expr: '{1!s}'".format(type(expr), expr))


def get_expr_shapes(*args, get_max_dim=False):
    if len(args) < 1:
        return None

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, dict):
            shape_struct = StructDict()
            for expr_id, expr in arg.items():
                shape_struct[expr_id] = get_expr_shape(expr)
            return shape_struct
        else:
            return get_expr_shape(arg)
    else:
        shapes = []
        for arg in args:
            shapes.append(get_expr_shape(arg))
        if get_max_dim:
            return tuple(np.maximum(*shapes))
        else:
            return shapes


_mld_dim_map = {
    'nx': ('A', 'E1'),
    'nu': ('B1', 'E2'),
    'ndelta': ('B2', 'E3'),
    'nz': ('B3', 'E4'),
    'nomega': ('B4', 'E5'),
    'n_cons': ('d',)
}

class MldBase(SortedStructDict):
    _internal_names = []
    _internal_names_set = SortedStructDict._internal_names_set.union(_internal_names)

    def _init_std_attributes(self):
        _sdict = super(MldBase, self)
        self._sdict_setitem = _sdict.__setitem__

    def __setattr__(self, key, value):
        super(MldBase, self).__setattr__(key, value)

    def __setitem__(self, key, value):
        if key in self.keys():
            self.update(**{key: value})
        elif key in self._internal_names_set:
            object.__setattr__(self, key, value)
        else:
            raise KeyError("key:'{}' is not valid or does not exist".format(key))

    def __repr__(self):
        data_repr = pprint.pformat(dict(self.items()), indent=0)
        return "".join([type(self).__name__, '(\n', data_repr, ')'])


class MldVarInfo(MldBase):
    _internal_names = ['_mld_model', 'mld_model']
    _internal_names_set = MldBase._internal_names_set.union(_internal_names)
    _valid_var_types = ['c', 'b']

    _state_names = ['x', 'u', 'delta', 'z', 'omega']

    _state_dim_names = ["".join(['n', name]) for name in _state_names]
    _con_dim_name = ['n_cons']
    _var_type_names = ["".join(['var_type_', name]) for name in _state_names]
    _bin_dim_names = ["".join([dim_name, '_l']) for dim_name in _state_dim_names]
    _allowed_data_set = set(_state_dim_names + _con_dim_name + _var_type_names + _bin_dim_names)

    def __init__(self, mld_model=None, bin_dims_struct=None, var_types_struct=None, **kwargs):

        if isinstance(mld_model, MldVarInfo):
            super(MldVarInfo, self).__init__(mld_model.items())
            self._mld_model = mld_model.mld_model
        else:
            super(MldVarInfo, self).__init__()
            super(MldVarInfo, self).update(dict.fromkeys(self._allowed_data_set))
            self._mld_model = None

        self.update(bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct, **kwargs)

    @property
    def mld_model(self):
        return self._mld_model

    @mld_model.setter
    def mld_model(self, value):
        raise AttributeError("Cannot set mld_model directly use MldVarInfo.update() instead.")

    def update(self, mld_model=None, bin_dims_struct=None, var_types_struct=None, **kwargs):
        if kwargs:
            bin_dims_struct, var_types_struct = self._update_set_struct_from_kwargs(
                bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct, kwargs=kwargs)
        if mld_model:
            self._mld_model = mld_model

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

            if (bin_dim or var_type is not None) and state_dim_name in ('ndelta', 'nz'):
                raise ValueError(
                    "Cannot manually set ndelta_l, nz_l, or associated var types these are fixed by the MLD "
                    "dimension.")
            elif state_dim_name == 'ndelta':
                bin_dim = self[state_dim_name]
                _temp_data[bin_dim_name] = bin_dim
            elif state_dim_name == 'nz':
                bin_dim = 0
                _temp_data[bin_dim_name] = bin_dim
            else:
                bin_dim = bin_dim or self._get_num_var_bin(var_type) or self.get(bin_dim_name) or 0
                _temp_data[bin_dim_name] = bin_dim

            if var_type is not None:
                var_type = self._check_var_types_vect_valid(var_type)
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

            super(MldVarInfo, self).update(_temp_data)

    def _check_var_types_vect_valid(self, var_types_vect):
        if var_types_vect is None:
            return var_types_vect
        else:
            var_types_vect = np.ravel(var_types_vect)
            if not np.setdiff1d(var_types_vect, self._valid_var_types).size == 0:
                raise ValueError('All elements of var_type_vectors must be in {}'.format(self._valid_var_types))
            return var_types_vect

    def _set_var_dims(self):
        for state_dim_name, sys_matrix_ids in _mld_dim_map.items():
            system_matrices = (self.mld_model.get(sys_id) for sys_id in sys_matrix_ids)
            if state_dim_name != 'n_cons':
                self._sdict_setitem(state_dim_name, get_expr_shapes(*system_matrices, get_max_dim=True)[1])
            else:
                self._sdict_setitem(state_dim_name, get_expr_shapes(*system_matrices, get_max_dim=True)[0])

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
        var_types_vect_flat = np.ravel(var_types_vect)
        return (var_types_vect_flat == 'b').sum()


class MldModel(MldBase):
    _internal_names = ['mld_info']
    _internal_names_set = MldBase._internal_names_set.union(_internal_names)
    _state_matrix_names = ['A', 'B1', 'B2', 'B3', 'B4', 'b5']
    _con_matrix_names = ['E1', 'E2', 'E3', 'E4', 'E5', 'd']
    _allowed_data_set = set(_state_matrix_names + _con_matrix_names)

    def __init__(self, system_matrices=None, bin_dims_struct=None, var_types_struct=None, **kwargs):
        if isinstance(system_matrices, MldModel):
                super(MldModel, self).__init__(system_matrices)
                self.mld_info = system_matrices.mld_info or MldVarInfo()
                system_matrices = dict(system_matrices.items())
        else:
            super(MldModel, self).__init__()
            super(MldModel, self).update(dict.fromkeys(self._allowed_data_set, np.empty(shape=(0, 0))))
            self.mld_info = MldVarInfo()  # initialize empty mld_var_info_struct

        self.update(system_matrices=system_matrices, bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct,
                    **kwargs)

    def update(self, system_matrices=None, bin_dims_struct=None, var_types_struct=None, **kwargs):

        if kwargs:
            bin_dims_struct, var_types_struct = self.mld_info._update_set_struct_from_kwargs(
                bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct, kwargs=kwargs)

        if system_matrices and kwargs:
            raise ValueError("Individual matrix arguments cannot be set if 'system_matrices' argument is set")
        creation_matrices = system_matrices or kwargs

        try:
            _temp_data = StructDict(self.items())
            for sys_matrix_id, system_matrix in creation_matrices.items():
                if sys_matrix_id in self._allowed_data_set:
                    old_val = self.get(sys_matrix_id)
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
            # if successful update real data
            super(MldModel, self).update(_temp_data)
        except ValueError as ve:
            raise ve

        self.mld_info.update(self, bin_dims_struct=bin_dims_struct, var_types_struct=var_types_struct)

    def verify_shapes_valid(self, mld_data):
        shapes_struct = get_expr_shapes(mld_data)

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

            if con_matrix_shape[0] not in (d_shape[0], 0):
                raise ValueError(
                    "Invalid shape for constraint matrix:'{0}':{1}, must have same row dimension as constraint "
                    "vector 'd', i.e. '({2},*)')".format(con_matrix_id, con_matrix_shape, d_shape[0]))
            if con_matrix_shape[1] not in (state_matrix_shape[1], 0) and con_matrix_id != 'd':
                raise ValueError(
                    "Invalid shape for constraint matrix:'{0}':{1}, must have same column dimension as state matrix "
                    "'{2}', i.e. '({3},*)')".format(con_matrix_id, con_matrix_shape, state_matrix_id,
                                                    state_matrix_shape[1]))

        for vect_id in ('b5', 'd'):
            shape_vect = shapes_struct[vect_id]
            if not np.isin(shape_vect, (0, 1)).any():
                raise ValueError(
                    "'{0}' must be of type vector, scalar or null array, currently has shape:{1}".format(vect_id,
                                                                                                         shape_vect))
        return shapes_struct

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




if __name__ == '__main__':
    a = dict.fromkeys(MldModel._allowed_data_set, np.ones((2, 2)))
    a.update(d=np.ones((2, 1)), b5=np.ones((2, 1)))
    mld = MldModel(a)
    mld2 = MldModel({'A': 1})
