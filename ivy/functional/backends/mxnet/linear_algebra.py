# global
import mxnet as mx
import numpy as _np

from collections import namedtuple
from mxnet.ndarray.ndarray import NDArray
from typing import Union, Optional, Tuple, Literal
from ivy.functional.backends.mxnet.old.general import matmul as _matmul


# local
from ivy import inf
import ivy as _ivy
DET_THRESHOLD = 1e-12

# Array API Standard #
# -------------------#

inv = mx.nd.linalg_inverse
cholesky = lambda x: mx.np.linalg.cholesky(x.as_np_ndarray()).as_nd_ndarray()


def pinv(x):
    """
    reference: https://help.matheass.eu/en/Pseudoinverse.html
    """
    x_dim, y_dim = x.shape[-2:]
    if x_dim == y_dim and mx.nd.sum(mx.nd.linalg.det(x) > DET_THRESHOLD) > 0:
        return inv(x)
    else:
        xT = mx.nd.swapaxes(x, -1, -2)
        xT_x = _ivy.to_native(_matmul(xT, x))
        if mx.nd.linalg.det(xT_x) > DET_THRESHOLD:
            return _matmul(inv(xT_x), xT)
        else:
            x_xT = _ivy.to_native(_matmul(x, xT))
            if mx.nd.linalg.det(x_xT) > DET_THRESHOLD:
                return _matmul(xT, inv(x_xT))
            else:
                return xT


def vector_norm(x: NDArray,
                p: Union[int, float, Literal[inf, - inf]] = 2,
                axis: Optional[Union[int, Tuple[int]]] = None,
                keepdims: bool = False) -> NDArray:
                
    return mx.np.linalg.norm(x,p,axis,keepdims)


def matrix_norm(x, p=2, axes=None, keepdims=False):
    axes = (-2, -1) if axes is None else axes
    if isinstance(axes, int):
        raise Exception('if specified, axes must be a length-2 sequence of ints,'
                        'but found {} of type {}'.format(axes, type(axes)))
    return mx.nd.norm(x, p, axes, keepdims=keepdims)


# noinspection PyPep8Naming
def svd(x: NDArray, full_matrices: bool = True) -> Union[NDArray, Tuple[NDArray,...]]:
    results=namedtuple("svd", "U S Vh")
    U, D, VT=_np.linalg.svd(x, full_matrices=full_matrices)
    res=results(U, D, VT)
    return res

    return mx.np.linalg.norm(x, p, axis, keepdims)


def diagonal(x: NDArray,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> NDArray:
    return diag(x, k=offset, axis1=axis1, axis2=axis2)

def slogdet(x: Union[_ivy.Array,_ivy.NativeArray],
            full_matrices: bool = True) -> Union[_ivy.Array, Tuple[_ivy.Array,...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = mx.linalg.slogdet(x)
    res = results(sign, logabsdet)
    
    return res

def trace(x: NDArray,
          offset: int = 0)\
              -> mx.np.ndarray:
    return mx.np.trace(x, offset=offset)


def qr(x, mode):
    return mx.np.linalg.qr(x, mode=mode)


# Extra #
# ------#

def vector_to_skew_symmetric_matrix(vector: NDArray)\
        -> NDArray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = mx.nd.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = mx.nd.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = mx.nd.concat(*(zs, -a3s, a2s), dim=-1)
    row2 = mx.nd.concat(*(a3s, zs, -a1s), dim=-1)
    row3 = mx.nd.concat(*(-a2s, a1s, zs), dim=-1)
    # BS x 3 x 3
    return mx.nd.concat(*(row1, row2, row3), dim=-2)
