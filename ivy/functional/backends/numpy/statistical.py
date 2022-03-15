# global
import numpy as _np
from typing import Tuple, Union


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


def prod(x: _np.ndarray,
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[_np.dtype] = None,
         keepdims: bool = False)\
        -> _np.ndarray:

    if dtype == None and _np.issubdtype(x.dtype,_np.integer):
        if _np.issubdtype(x.dtype,_np.signedinteger) and x.dtype in [_np.int8,_np.int16,_np.int32]:
            dtype = _np.int32
        elif _np.issubdtype(x.dtype,_np.unsignedinteger) and x.dtype in [_np.uint8,_np.uint16,_np.uint32]:
            dtype = _np.uint32
        elif x.dtype == _np.int64: 
            dtype = _np.int64
        else:
            dtype = _np.uint64

    return _np.prod(a=x,axis=axis,dtype=dtype,keepdims=keepdims)

