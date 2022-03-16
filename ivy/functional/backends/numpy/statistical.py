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


def var(x: _np.ndarray,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> _np.ndarray:
    return _np.var(_np.array_api.asarray(x), axis=axis, keepdims=keepdims)
