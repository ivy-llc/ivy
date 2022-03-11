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

def slogdet(x:Union[_ivy.Array,_ivy.NativeArray],full_matrices: bool = True) -> Union[_ivy.Array, Tuple[_ivy.Array,...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = mx.linalg.slogdet(x)
    res = results(sign, logabsdet)
    return res
