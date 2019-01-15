from utils.structdict import StructDict

import numpy as np
from numpy.lib.stride_tricks import as_strided as _as_strided
import scipy.linalg as scl
import scipy.sparse as scs

from collections import namedtuple as NamedTuple

from utils.decorator_utils import cache_hashable_args


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
    # used to ensure block toeplitz is compatible with scipy.linalg.toeplitz
    arr = np.asanyarray(arr, dtype=dtype, order=order)
    if arr.ndim == 0:
        result = arr.reshape(1,1,1)
    elif arr.ndim == 1:
        result = arr[:, np.newaxis, np.newaxis]
    elif arr.ndim == 2:
        result = arr[:, np.newaxis]
    else:
        result = arr
    return result

def block_diag_dense_same_shape(mats, format=None, dtype=None):
    arrs = np.array(mats)
    arrs = _atleast_3d_col(arrs)
    k, n, m = arrs.shape
    vals = np.concatenate([arrs, np.zeros(shape=(k, n, (k - 1) * m), dtype=arrs.dtype)], axis=2)

    item_size = arrs.itemsize
    shape = (k, n, k * m)
    strides = ((k * n - 1) * m * item_size, k * m * item_size, item_size)
    block_diag = _as_strided(vals, shape=shape, strides=strides).reshape(n * k, m * k)
    if dtype is not None:
        block_diag = block_diag.astype(dtype)
    return block_diag.copy()


def block_diag_dense(mats, format=None, dtype=None):
    a_mats = np.asarray(mats)
    if a_mats.dtype != np.object:
        block_diag = block_diag_dense_same_shape(a_mats, format=format, dtype=dtype)
    else:
        block_diag = scl.block_diag(*mats)

    if dtype is not None:
        block_diag = block_diag.astype(dtype)
    return block_diag


def create_object_array(tup):
    try:
        obj_arr = np.empty(len(tup), dtype='object')
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

    np_toep_strided = _as_strided(vals[c.shape[0] - 1:], shape=stride_shp, strides=(-n, m, n, k)).reshape(out_shp)

    if sparse:
        if np_toep_strided.dtype != np.object_:
            return scs.csr_matrix(np_toep_strided)
        elif all(isinstance(block, scs.csr_matrix) for block in np_toep_strided.flat):
            v_stacked = [scs.bmat(np.atleast_2d(col).T).tocsc() for col in np_toep_strided.T]
            return scs.bmat(np.atleast_2d(v_stacked)).tocsr()
        else:
            h_stacked = [scs.bmat(np.atleast_2d(row)).tocsr() for row in np_toep_strided]
            return scs.bmat(np.atleast_2d(h_stacked).T).tocsc()
    else:
        return np_toep_strided.copy()


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
                'zeros']

_MatOpsNameTup = NamedTuple('MatOps', _MatOpsNames)


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
            zeros=scs.csr_matrix
        )
    else:
        mat_ops = _MatOpsNameTup(
            package=np,
            linalg=np.linalg,
            sclinalg=scl,
            block_diag=block_diag_dense,
            vmatrix=np.atleast_2d,
            hmatrix=np.atleast_2d,
            zeros=np.zeros
        )
    return mat_ops


mats = [np.arange(1, 7).reshape(3, 2).astype(float) + 6 * i for i in range(1, 100)]


