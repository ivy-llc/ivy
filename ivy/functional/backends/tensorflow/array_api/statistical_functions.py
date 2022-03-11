# global
_round = round
import tensorflow as _tf
from tensorflow.experimental import numpy as _np
from typing import Tuple, Union, Optional

def min(x: _tf.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _tf.Tensor:
    return _tf.math.reduce_min(x, axis = axis, keepdims = keepdims)

def prod(x: _tf.Tensor,
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[_tf.DType] = None,
         keepdims: bool = False)\
        -> _tf.Tensor:


    if x.dtype in [_np.int8,_np.int16,_np.int32]:
        dtype = _np.int32
    elif x.dtype in [_np.uint8,_np.uint16,_np.uint32]:
        dtype = _np.uint32
    elif x.dtype == _np.int64: 
        dtype = _np.int64
    else:
        dtype = _np.uint64  

    return _tf.experimental.numpy.prod(x,axis,dtype,keepdims)