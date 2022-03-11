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

    return _tf.experimental.numpy.prod(x,axis,dtype,keepdims)