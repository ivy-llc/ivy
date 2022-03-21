# global
_round = round
import tensorflow as _tf
from typing import Tuple, Union, Optional


def min(x: _tf.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _tf.Tensor:
    return _tf.math.reduce_min(x, axis = axis, keepdims = keepdims)

def max(x: _tf.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _tf.Tensor:
    return _tf.math.reduce_max(x, axis = axis, keepdims = keepdims)

def var(x: _tf.Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> _tf.Tensor:
    m = _tf.reduce_mean(x, axis=axis, keepdims=True)
    return _tf.reduce_mean(_tf.square(x - m), axis=axis, keepdims=keepdims)
