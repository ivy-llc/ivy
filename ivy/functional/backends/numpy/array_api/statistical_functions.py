# global
import numpy as _np
from typing import Tuple, Union


def min(x: _np.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _np.ndarray:
    return _np.amin(a = x, axis = axis, keepdims = keepdims)

def prod(x: _np.ndarray,
         axis: Union[int, Tuple[int]] = None,
         dtype: _np.dtype = None,
         keepdims: bool = False)\
        -> _np.ndarray:
    return _np.prod(x,axis,dtype,keepdims)