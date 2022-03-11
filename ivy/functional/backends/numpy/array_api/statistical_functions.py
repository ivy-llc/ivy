# global
import numpy as _np
from typing import Tuple, Union

# local
from ivy import dtype_from_str, default_dtype

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
    if dtype == None and _np.issubdtype(x.dtype, _np.signedinteger):
        dtype = _np.int32
    elif dtype == None and  _np.issubdtype(x.dtype, _np.unsignedinteger):
        dtype = _np.uint32
    return _np.prod(a=x,axis=axis,dtype=dtype,keepdims=keepdims)