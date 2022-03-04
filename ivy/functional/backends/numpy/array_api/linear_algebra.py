# global
import numpy as np
from typing import Union, Optional, Tuple, Literal
from numpy import asarray_chkfinite, asarray, atleast_2d
# local
from ivy import inf


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


def diagonal(x: np.ndarray,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> np.ndarray:
    return np.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def cholesky(x: np.ndarray,
            upper: bool = False) -> np.ndarray:
    
    # a = asarray_chkfinite(x)
    # a = atleast_2d(a)

    # # Dimension check
    # if a.ndim != 2:
    #     raise ValueError('Input array needs to be 2D but received '
    #                      'a {}d-array.'.format(a.ndim))
    # # Squareness check
    # if a.shape[0] != a.shape[1]:
    #     raise ValueError('Input array is expected to be square but has '
    #                      'the shape: {}.'.format(a.shape))

    # # Quick return for square empty array
    # if a.size == 0:
    #     return a.copy()

    if not upper:
        return np.linalg.cholesky(x)
    else:
        return np.linalg.cholesky(x).T.conj()