import numpy as np
import numpy.array_api as npa
from typing import Union, Tuple, Optional, List


def var(x: np.ndarray,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) -> np.ndarray:
    return np.var(npa.asarray(x), axis=axis, keepdims=keepdims)


def min(x: np.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> np.ndarray:
    return np.amin(a = x, axis = axis, keepdims = keepdims)
