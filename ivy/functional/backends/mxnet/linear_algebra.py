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

inv = mx.nd.linalg_inverse
cholesky = lambda x: mx.np.linalg.cholesky(x.as_np_ndarray()).as_nd_ndarray()
DET_THRESHOLD = 1e-12


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
