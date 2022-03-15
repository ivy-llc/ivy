# global
import numpy as np
from typing import Union, Optional, Tuple, Literal
from collections import namedtuple

# local
from ivy import inf
import ivy
from collections import namedtuple


# noinspection PyUnusedLocal,PyShadowingBuiltins
def vector_norm(x: np.ndarray,
                axis: Optional[Union[int, Tuple[int]]] = None,
                keepdims: bool = False,
                ord: Union[int, float, Literal[inf, - inf]] = 2)\
                 -> np.ndarray:

    if axis is None:
        np_normalized_vector = np.linalg.norm(x.flatten(), ord, axis, keepdims)

    else:
        np_normalized_vector = np.linalg.norm(x, ord, axis, keepdims)

    if np_normalized_vector.shape == tuple():
        return np.expand_dims(np_normalized_vector, 0)
    return np_normalized_vector


def svd(x:np.ndarray,full_matrices: bool = True) -> Union[np.ndarray, Tuple[np.ndarray,...]]:
    results=namedtuple("svd", "U S Vh")
    U, D, VT=np.linalg.svd(x, full_matrices=full_matrices)
    res=results(U, D, VT)
    return res
  
def diagonal(x: np.ndarray,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> np.ndarray:
    return np.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def qr(x: np.ndarray,
       mode: str = 'reduced') -> namedtuple('qr', ['Q', 'R']):
    res = namedtuple('qr', ['Q', 'R'])
    q, r = np.linalg.qr(x, mode=mode)
    return res(q, r)

  
def matmul(x1: np.ndarray,
           x2: np.ndarray) -> np.ndarray:
    return np.matmul(x1, x2)

  
def slogdet(x:Union[ivy.Array,ivy.NativeArray],full_matrices: bool = True) -> Union[ivy.Array, Tuple[ivy.Array,...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = np.linalg.slogdet(x)
    res = results(sign, logabsdet)
    return res

def trace(x: np.ndarray, 
          offset: int = 0)\
              -> np.ndarray:
    return np.trace(x, offset)
