# global
import numpy as np
from typing import Union, Optional, Tuple, Literal
import numpy.array_api as npa
# local
import ivy
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


def cross(x1: np.ndarray, x2: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.cross(x1, x2, axis=axis)
