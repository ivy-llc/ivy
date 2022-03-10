# global
import mxnet as mx
import numpy as _np
from typing  import Union, Optional, Tuple, Literal
from collections import namedtuple



# local
from ivy import inf
import ivy as _ivy
 
  
def vector_norm(x: mx.ndarray.ndarray.NDArray,
                p: Union[int, float, Literal[inf, - inf]] = 2,
                axis: Optional[Union[int, Tuple[int]]] = None,
                keepdims: bool = False)\
                    -> mx.ndarray.ndarray.NDArray:
                
    return mx.np.linalg.norm(x,p,axis,keepdims)

# noinspection PyPep8Naming
def svd(x:mx.ndarray.ndarray.NDArray,full_matrices: bool = True) -> Union[mx.ndarray.ndarray.NDArray, Tuple[mx.ndarray.ndarray.NDArray,...]]:
    results=namedtuple("svd", "U S Vh")
    U, D, VT=_np.linalg.svd(x, full_matrices=full_matrices)
    res=results(U, D, VT)
    return res

    return mx.np.linalg.norm(x, p, axis, keepdims)


def diagonal(x: mx.nd.NDArray,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> mx.nd.NDArray:
    return mx.nd.diag(x, k=offset, axis1=axis1, axis2=axis2)


def cross(x1: mx.ndarray.ndarray.NDArray, x2: mx.ndarray.ndarray.NDArray) -> mx.ndarray.ndarray.NDArray:
    a1 = x1[..., 0:1]
    a2 = x1[..., 1:2]
    a3 = x1[..., 2:3]
    b1 = x2[..., 0:1]
    b2 = x2[..., 1:2]
    b3 = x2[..., 2:3]
    res1 = a2 * b3 - a3 * b2
    res2 = a3 * b1 - a1 * b3
    res3 = a1 * b2 - a2 * b1
    res = mx.nd.concat(res1, res2, res3, dim=-1)
    return res