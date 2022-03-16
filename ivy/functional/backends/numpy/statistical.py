# global
import numpy as _np
from typing import Tuple, Union, Optional


def min(x: _np.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _np.ndarray:
    return _np.amin(a=x, axis=axis, keepdims=keepdims)


def max(x: _np.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _np.ndarray:
    return _np.amax(a=x, axis=axis, keepdims=keepdims)


def var(x: np.ndarray,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> np.ndarray:
    return np.var(npa.asarray(x), axis=axis, keepdims=keepdims)
