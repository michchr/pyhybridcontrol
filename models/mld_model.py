from utils.structdict import SortedStructDict, StructDict

import pprint
import numpy as np
from reprlib import recursive_repr
import copy
import sympy as sp
import numbers

from collections import UserDict


# def append_named_call_args(func):
#     def wrapper(self, *args, **kwargs):
#         named_args = inspect.getfullargspec(func).args
#         return func(self, *args, **kwargs, named_call_args=named_args)
#
#     wrapper.__signature__ = inspect.signature(func)
#     return wrapper

def get_shape(expr):
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
                shape_struct[expr_id] = get_shape(expr)
            return shape_struct
        else:
            return get_shape(arg)
    else:
        shapes = []
        for arg in args:
            shapes.append(get_shape(arg))
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


class MldVarInfoStruct(StructDict):
    _valid_var_types = ['c', 'b']

    _computed_vars = ['nx', 'nu', 'ndelta', 'nz', 'nomega', 'n_cons', 'ndelta_l', 'nz_l', 'delta_var_types',
                      'z_var_types']

    _user_vars = ['nx_l', 'nu_l', 'nomega_l', 'x_var_types', 'u_var_types', 'omega_var_types']

    def __init__(self, mld_model=None, nx_l=None, nu_l=None,
                 nomega_l=None, x_var_types=None, u_var_types=None, omega_var_types=None):

        if mld_model is None:
            mld_model = MldModel()
        self._mld_model = mld_model

        super(MldVarInfoStruct, self).__init__()
        super(MldVarInfoStruct, self).update(dict.fromkeys(self._computed_vars))
        super(MldVarInfoStruct, self).update(dict.fromkeys(self._user_vars))

        self._set_var_dims()
        # self.set_var_info(nx_l=nx_l, nu_l=nu_l, nomega_l=nomega_l, x_var_types=x_var_types,
        #                  u_var_types=u_var_types, omega_var_types=omega_var_types)

    @property
    def mld_model(self):
        return self._mld_model

    @mld_model.setter
    def mld_model(self, mld_model):
        self.__init__(mld_model=mld_model)

    def __setitem__(self, key, value):
        if key in self._user_vars:
            self.set_var_info(**{key: value})
        else:
            super(MldVarInfoStruct, self).__setitem__(key, value)

    def set_var_info(self, nx_l=None, nu_l=None, nomega_l=None, x_var_types=None, u_var_types=None,
                     omega_var_types=None):

        _set_direct = super(MldVarInfoStruct, self).__setitem__

        # either set to new value, old value, or zero in that order - never None
        _set_direct('nx_l', nx_l or self._get_num_var_bin(x_var_types) or self.get('nx_l') or 0)
        _set_direct('nu_l', nu_l or self._get_num_var_bin(u_var_types) or self.get('nu_l') or 0)
        _set_direct('nomega_l', nomega_l or self._get_num_var_bin(omega_var_types) or self.get('nomega_l') or 0)
        _set_direct('ndelta_l', self.ndelta)  # all delta are always binary
        _set_direct('nz_l', 0)  # all z are always continous

        _set_direct('x_var_types', np.ravel(x_var_types) or np.hstack(
            [np.repeat('c', self.nx - self.nx_l), np.repeat('b', self.nx_l)]))
        _set_direct('u_var_types', np.ravel(u_var_types) or np.hstack(
            [np.repeat('c', self.nu - self.nu_l), np.repeat('b', self.nu_l)]))
        _set_direct('omega_var_types', np.ravel(omega_var_types) or np.hstack(
            [np.repeat('c', self.nomega - self.nomega_l), np.repeat('b', self.nomega_l)]))

    def _check_var_types_vect_valid(self, var_types_vect):
        if var_types_vect is None:
            return var_types_vect
        else:
            var_types_vect = np.asarray(var_types_vect)
            if not np.setdiff1d(var_types_vect, self._valid_var_types).size == 0:
                raise ValueError('All elements of var_types_vectors must be in {}'.format(self._valid_var_types))
            return var_types_vect

    def _set_var_dims(self):
        for dim_id, sys_matrix_ids in _mld_dim_map.items():
            system_matrices = (self.mld_model[sys_id] for sys_id in sys_matrix_ids)
            if dim_id != 'n_cons':
                self[dim_id] = get_expr_shapes(*system_matrices, get_max_dim=True)[1]
            else:
                self[dim_id] = get_expr_shapes(*system_matrices, get_max_dim=True)[1]

    def __repr__(self):
        _dict = dict(self)
        data_repr = pprint.pformat(_dict)
        return "".join([type(self).__name__, '(\n', data_repr, ')'])

    @staticmethod
    def _get_num_var_bin(var_types_vect):
        if var_types_vect is None:
            return None
        var_types_vect_flat = np.ravel(var_types_vect)
        return (var_types_vect_flat == 'b').sum()


class MldModel:
    _internal_names = ['_data', 'mld_var_info_struct']
    _internal_names_set = set(_internal_names)
    _state_matrix_names = ['A', 'B1', 'B2', 'B3', 'B4', 'b5']
    _con_matrix_names = ['E1', 'E2', 'E3', 'E4', 'E5', 'd']
    _allowed_system_matrices_set = set(_state_matrix_names).union(_con_matrix_names)

    def __init__(self, system_matrices=None,
                 nx_l=None, nu_l=None, nomega_l=None,
                 x_var_types=None, u_var_types=None,
                 omega_var_types=None, **kwargs):

        self._data = SortedStructDict(dict.fromkeys(self._allowed_system_matrices_set, np.empty(shape=(0, 0))))
        self.update_matrices(system_matrices=system_matrices,
                             nx_l=nx_l, nu_l=nu_l, nomega_l=nomega_l,
                             x_var_types=x_var_types, u_var_types=u_var_types,
                             omega_var_types=omega_var_types, **kwargs)

        self.mld_var_info_struct = MldVarInfoStruct(self)

    def update_matrices(self, system_matrices=None,
                        nx_l=None, nu_l=None, nomega_l=None,
                        x_var_types=None, u_var_types=None,
                        omega_var_types=None, **kwargs):

        if system_matrices and kwargs:
            raise ValueError("Individual matrix arguments cannot be set if 'system_matrices' argument is set")
        creation_matrices = system_matrices or kwargs

        if creation_matrices:
            try:
                _temp_data = self._data.copy()  # create shallow copy
                for sys_matrix_id, system_matrix in creation_matrices.items():
                    if sys_matrix_id in self._allowed_system_matrices_set:
                        old_val = self._data.get(sys_matrix_id)
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
                self._data = _temp_data
            except ValueError as ve:
                raise ve

    def verify_shapes_valid(self, mld_data):
        shapes_struct = get_expr_shapes(mld_data)

        A_shape = shapes_struct.A
        d_shape = shapes_struct.d

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

    def __setattr__(self, key, value):

        # if attribute exists set and return
        try:
            object.__getattribute__(self, key)
            return object.__setattr__(self, key, value)
        except AttributeError:
            pass

        if key in self._internal_names_set:
            return super(MldModel, self).__setattr__(key, value)

        if key in self._data:
            return self.__setitem__(key, value)
        else:
            super(MldModel, self).__setattr__(key, value)

    def __setitem__(self, key, value):
        self._data[key] = value
        self.mld_var_info = MldVarInfoStruct(self)

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, key):
        if key in self._internal_names_set:
            try:
                return object.__getattribute__(self, key)
            except AttributeError as e:
                raise e
        try:
            return self._data.__getattr__(key)
        except AttributeError:
            raise AttributeError("Attribute/system matrix with name: '{}' does not exist".format(key))

    def __repr__(self):
        # data_repr = self._data.__repr__().split('(', 1)[-1][:-1]
        # return "".join([type(self).__name__, '(', data_repr, ')'])
        data_repr = pprint.pformat(self._data)
        return "".join([type(self).__name__, '(\n', data_repr, ')'])

    def __dir__(self):
        orig_dir = set(dir(type(self)))
        __dict__keys = set(self.__dict__.keys())
        additions = set(self._data.keys())
        rv = orig_dir | __dict__keys | additions
        return sorted(rv)


if __name__ == '__main__':
    mld = MldModel(A=np.ones((2, 2)), B1=np.ones((2, 4)), B2=np.ones((2, 1)), E1=np.ones((10, 2)), d=np.ones((10, 1)),
                   b5=np.ones((2, 0)))
    mld2 = MldModel({'A': 1})
