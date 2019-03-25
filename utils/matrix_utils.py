import inspect

import sympy as sp
import wrapt

from utils.func_utils import get_cached_func_spec, make_function
from structdict import StructDict, OrderedStructDict

import numpy as np
from numpy.lib.stride_tricks import as_strided as _as_strided
import scipy.linalg as scl
import scipy.sparse as scs
from collections import namedtuple as NamedTuple

from utils.decorator_utils import cache_hashable_args
import functools

def is_scalar_like(val):
    shape = getattr(val, 'shape', (1,))
    return all([d==1 for d in shape])

def matmul(self, other):
    if any(map(is_scalar_like, (self, other))):
        return self * other
    else:
        return self @ other


def atleast_2d_col(arr, dtype=None, order=None):
    arr = np.asanyarray(arr, dtype=dtype, order=order)
    if arr.ndim == 0:
        result = arr.reshape(1, 1)
    elif arr.ndim == 1:
        result = arr[:, np.newaxis]
    else:
        result = arr
    return result


def _atleast_3d_col(arr, dtype=None, order=None):
    arr = np.asanyarray(arr, dtype=dtype, order=order)
    if arr.ndim == 0:
        result = arr.reshape(1, 1, 1)
    elif arr.ndim == 1:
        result = arr[:, np.newaxis, np.newaxis]
    elif arr.ndim == 2:
        result = arr[np.newaxis, :]
    else:
        result = arr
    return result


def block_diag_dense_same_shape(mats, format=None, dtype=None):
    arrs = _atleast_3d_col(mats, dtype=dtype)
    k, n, m = arrs.shape
    arrs = arrs.reshape(k * n, m)
    vals = np.zeros(shape=(k * n, k * m), dtype=arrs.dtype)
    vals[:, :m] = arrs

    item_size = arrs.itemsize
    shape = (k, n, k * m)
    strides = ((k * n - 1) * m * item_size, k * m * item_size, item_size)
    strided = np.ascontiguousarray(_as_strided(vals, shape=shape, strides=strides))

    block_diag = strided.reshape(n * k, m * k)
    return block_diag


def block_diag_dense(mats, format=None, dtype=None):
    # scl.blockdiag is faster for large matrices or a large number of matrices.
    a_mats = _atleast_3d_col(mats)
    if a_mats.dtype != np.object_ and np.prod(a_mats.shape) < 720:
        block_diag = block_diag_dense_same_shape(a_mats, format=format, dtype=dtype)
    else:
        block_diag = scl.block_diag(*a_mats)

    if dtype is not None:
        block_diag = block_diag.astype(dtype)
    return block_diag


import timeit


def block_diag_test(a, number=1000):
    def t1():
        return block_diag_dense(a)

    def t2():
        return scl.block_diag(*a)

    tt1 = timeit.timeit("t1()", globals=locals(), number=number)
    print("block_diag_dense", tt1)
    tt2 = timeit.timeit("t2()", globals=locals(), number=number)
    print("scl.block_diag", tt2)

    t1 = t1()
    t2 = t2()
    print("t1", t1.dtype)
    print("t2", t2.dtype)
    return np.array_equal(t1, t2)


def create_object_array(tup):
    try:
        obj_arr = np.empty(len(tup), dtype=np.object_)
    except TypeError:
        raise TypeError("tup must be array like.")

    for ind, item in enumerate(tup):
        obj_arr[ind] = item
    return obj_arr


def block_toeplitz(c_tup, r_tup=None, sparse=False):
    """
     Based on scipy.linalg.toeplitz method but applied in a block fashion.
     """
    try:
        c = np.array(c_tup)
    except ValueError:
        c = create_object_array(c_tup)

    if r_tup is None:
        if np.issubdtype(c.dtype, np.number):
            r = c.conjugate()
        else:
            r = c
    else:
        try:
            r = np.array(r_tup)
        except ValueError:
            r = create_object_array(r_tup)

    c = _atleast_3d_col(c)
    r = _atleast_3d_col(r)
    # # Form a array containing a reversed c followed by r[1:] that could be strided to give us a toeplitz matrix.
    try:
        vals = np.concatenate((c[::-1], r[1:]))
    except ValueError as ve:
        raise ValueError("Incompatible dimensions in c_tup or between c_tup and r_tup - " + ve.args[0])
    stride_shp = (c.shape[0], c.shape[1], r.shape[0], r.shape[2])
    out_shp = (c.shape[0] * c.shape[1], r.shape[0] * r.shape[2])
    n, m, k = vals.strides
    strided = np.ascontiguousarray(_as_strided(vals[c.shape[0] - 1:], shape=stride_shp, strides=(-n, m, n, k)))

    np_toeplitz = strided.reshape(out_shp)

    if sparse:
        if np_toeplitz.dtype != np.object_:
            return scs.csr_matrix(np_toeplitz)
        elif all(isinstance(block, scs.csr_matrix) for block in np_toeplitz.flat):
            v_stacked = [scs.bmat(np.atleast_2d(col).T).tocsc() for col in np_toeplitz.T]
            return scs.bmat(np.atleast_2d(v_stacked)).tocsr()
        else:
            h_stacked = [scs.bmat(np.atleast_2d(row)).tocsr() for row in np_toeplitz]
            return scs.bmat(np.atleast_2d(h_stacked).T).tocsc()
    else:
        return np_toeplitz


def block_toeplitz_alt(c_tup, r_tup=None, sparse=False):
    c = create_object_array(c_tup)
    if r_tup is None:
        try:
            r = c.conjugate()
        except AttributeError:
            r = c
    else:
        r = create_object_array(r_tup)
    # # Form a 1D array containing a reversed c followed by r[1:] that could be
    # # strided to give us toeplitz matrix.
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = c.shape[0], r.shape[0]
    n = vals.strides[0]
    strided = _as_strided(vals[len(c) - 1:], shape=out_shp, strides=(-n, n))
    np_toep = np.block(strided.tolist())

    if sparse:
        if all(isinstance(block, scs.csr_matrix) for block in np_toep.flat):
            v_stacked = [scs.bmat(np.atleast_2d(col).T).tocsc() for col in np_toep.T]
            return scs.bmat(np.atleast_2d(v_stacked)).tocsr()
        else:
            h_stacked = [scs.bmat(np.atleast_2d(row)).tocsr() for row in np_toep]
            return scs.bmat(np.atleast_2d(h_stacked).T).tocsc()
    else:
        return np_toep


_MatOpsNames = ['package',
                'linalg',
                'sclinalg',
                'block_diag',
                'vmatrix',
                'hmatrix',
                'zeros',
                'vstack',
                'hstack',
                'matmul']

_MatOpsNameTup = NamedTuple('MatOps', _MatOpsNames)


def pass_through(a):
    return a


@cache_hashable_args(maxsize=2)
def get_mat_ops(sparse=False):
    if sparse:
        mat_ops = _MatOpsNameTup(
            package=scs,
            linalg=scs,
            sclinalg=scs,
            block_diag=scs.block_diag,
            vmatrix=scs.csr_matrix,
            hmatrix=scs.csc_matrix,
            zeros=scs.csr_matrix,
            vstack=scs.vstack,
            hstack=scs.hstack,
            matmul=functools.partial(matmul, sparse=True)
        )
    else:
        mat_ops = _MatOpsNameTup(
            package=np,
            linalg=np.linalg,
            sclinalg=scl,
            block_diag=block_diag_dense,
            vmatrix=np.atleast_2d,
            hmatrix=np.atleast_2d,
            zeros=np.zeros,
            vstack=np.vstack,
            hstack=np.hstack,
            matmul=matmul
        )
    return mat_ops


def get_expr_shape(expr):
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


def get_expr_shapes(*exprs, get_max_dim=False):
    if not exprs:
        return None

    if isinstance(exprs[0], dict):
        shapes = StructDict({expr_id: get_expr_shape(expr) for expr_id, expr in exprs[0].items()})
    else:
        shapes = [get_expr_shape(expr) for expr in exprs]

    if get_max_dim:
        shapes = list(shapes.values()) if isinstance(shapes, dict) else shapes
        return tuple(np.maximum.reduce(shapes))
    else:
        return shapes


class CallableMatrixMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        if kwargs.pop('_create_object', False):
            obj.__init__(*args, **kwargs)
        return obj


class CallableMatrixBase(metaclass=CallableMatrixMeta):

    def __new__(cls, matrix, matrix_name=None, _nan_call=None, _create_object=False):
        if _create_object:
            self = super(CallableMatrixBase, cls).__new__(cls, matrix, matrix_name, _nan_call=_nan_call)
            return self
        matrix_func = cls._process_matrix_func(matrix)
        nan_call = cls._nan_call(matrix_func)
        if np.all(np.isfinite(nan_call)):
            return CallableMatrixConstant(matrix, matrix_name, _nan_call=_nan_call, _create_object=True)
        else:
            return CallableMatrix(matrix, matrix_name, _nan_call=_nan_call, _create_object=True)

    @classmethod
    def _process_matrix_func(cls, matrix):
        def const_func(constant):
            def constant_func():
                return constant

            return constant_func

        if isinstance(matrix, (sp.Expr, sp.Matrix)):
            system_matrix = sp.Matrix(matrix)
            param_sym_tup = cls._get_param_sym_tup(system_matrix)
            func = sp.lambdify(param_sym_tup, system_matrix, modules="numpy", dummify=False)
        elif inspect.isfunction(matrix):
            func = matrix
        elif inspect.ismethod(matrix):
            func = matrix.__func__
        else:
            func = const_func(atleast_2d_col(matrix))

        return func

    @staticmethod
    def _nan_call(matrix_func):
        f_spec = get_cached_func_spec(matrix_func, reset_cache=True)
        kwargs = {param_name: np.NaN for param_name in f_spec.all_kw_params}
        args = [np.NaN] * len(f_spec.pos_only_params)

        try:
            ret_val = atleast_2d_col(matrix_func(*args, **kwargs))
            ret_val.setflags(write=False)
            return ret_val
        except TypeError:
            msg = f"_nan_call() failed, it is likely that the matrix function does not have a constant shape.\n"
            note = (
                "Note: all callable expressions must return with a constant array shape that does not depend on its "
                "arguments. Shape is determined by calling the function with all arguments set to a float with value "
                "NaN.")
            raise TypeError(msg + note)

    @staticmethod
    def _get_param_sym_tup(expr):
        try:
            sym_dict = {str(sym): sym for sym in expr.free_symbols}
            param_sym_tup = tuple([sym_dict.get(sym) for sym in sorted(sym_dict.keys())])
        except AttributeError:
            param_sym_tup = ()

        return param_sym_tup


class CallableMatrix(CallableMatrixBase, wrapt.decorators.AdapterWrapper):

    def __init__(self, matrix, matrix_name=None, **kwargs):
        if isinstance(matrix, type(self)):
            super(CallableMatrix, self).__init__(wrapped=matrix.__wrapped__, wrapper=matrix._self_wrapper, enabled=None,
                                                 adapter=matrix._self_adapter)
            nan_call = self._nan_call(matrix.__wrapped__)
        else:
            _nan_call = kwargs.get('_nan_call')
            if _nan_call:
                matrix_func = matrix
                nan_call = _nan_call
            else:
                matrix_func = type(self)._process_matrix_func(matrix)
                nan_call = self._nan_call(matrix_func)

            self._self_matrix_name = matrix_name if matrix_name is not None else matrix_func.__name__
            self._self_wrapped_name = matrix_func.__name__

            matrix_func.__name__ = self._self_matrix_name
            matrix_func.__qualname__ = (
                "".join(matrix_func.__qualname__.rsplit('.', 1)[:-1] + ['.', matrix_func.__name__]).lstrip('.'))
            self._self_wrapped_f_spec = get_cached_func_spec(matrix_func)

            adapter = self._gen_callable_matrix_adapter(self._self_wrapped_f_spec)
            self._self_adapter_spec = get_cached_func_spec(adapter, bypass_cache=True)

            wrapper = self._matrix_wrapper
            super(CallableMatrix, self).__init__(wrapped=matrix_func, wrapper=wrapper, enabled=None, adapter=adapter)

        try:
            self.__delattr__('_self_shape')
        except AttributeError:
            pass

        self._self_shape = get_expr_shape(nan_call)
        self._self_size = np.prod(nan_call.size)
        self._self_ndim = nan_call.ndim
        self._self_dtype = nan_call.dtype
        self._self_nbytes = nan_call.nbytes
        self._self_itemsize = nan_call.itemsize
        self._self_is_empty = False if self._self_size else True
        self._self_is_all_zero = np.all(nan_call == 0)
        self._self_is_constant = np.all(np.isfinite(nan_call))

        if self._self_is_constant:
            if type(self) == CallableMatrix:
                raise TypeError(f"Cannot initialize {type(self).__name__} object with constant matrix.")
            self._self_constant = nan_call
        else:
            self._self_constant = None

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
            msg = te.args[0].replace(self._self_wrapped_name, wrapped.__name__)
            raise TypeError(msg).with_traceback(te.__traceback__) from None

        if getattr(retval, 'ndim', 0) < 2:
            retval = atleast_2d_col(retval)

        if isinstance(retval, np.ndarray):
            retval.setflags(write=False)

        return retval

    def _gen_callable_matrix_adapter(self, f_spec):
        f_args_spec_struct = OrderedStructDict(f_spec.arg_spec._asdict()).deepcopy()
        f_args_spec_struct.kwonlyargs.append('param_struct')
        if f_args_spec_struct.kwonlydefaults:
            f_args_spec_struct.kwonlydefaults.update({'param_struct': None})
        else:
            f_args_spec_struct.kwonlydefaults = {'param_struct': None}

        f_args_spec = inspect.FullArgSpec(**f_args_spec_struct)
        adapter = make_function(f_args_spec, name='adapter')
        return adapter

    def __reduce__(self):
        return (type(self), (self.__wrapped__, self._self_matrix_name))

    @property
    def __name__(self):
        return self._self_matrix_name

    @property
    def __class__(self):
        return type(self)

    @property
    def _f_spec(self):
        return self._self_adapter_spec

    @_f_spec.setter
    def _f_spec(self, f_spec):
        self._self_adapter_spec = f_spec

    @property
    def __signature__(self):
        return self._self_adapter_spec.signature

    @property
    def required_params(self):
        return self._self_wrapped_f_spec.all_kw_params

    @property
    def matrix_name(self):
        return self._self_matrix_name

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

    @property
    def is_empty(self):
        return self._self_is_empty

    @property
    def is_all_zero(self):
        return self._self_is_all_zero

    @property
    def is_constant(self):
        return self._self_is_constant

    def __repr__(self):
        empty_str = f", shape={self._self_shape}" if not self._self_size else ""
        return f"<{self.__class__.__name__} {self.__name__}{self.__signature__}{empty_str}>"

    def __str__(self):
        return self.__repr__()

    def __dir__(self):
        wrapped_dir = set(dir(self.__wrapped__))
        added_dir = set(type(self).__dict__)
        rv = wrapped_dir | added_dir
        return sorted(rv)


class CallableMatrixConstant(CallableMatrix):

    def __init__(self, matrix, matrix_name=None, **kwargs):
        super(CallableMatrixConstant, self).__init__(matrix, matrix_name=matrix_name, **kwargs)
        if not self.is_constant:
            raise TypeError(f"Cannot initialize {type(self).__name__} object with non-constant matrix.")

    def __call__(self, *, param_struct=None):
        return self._self_constant
