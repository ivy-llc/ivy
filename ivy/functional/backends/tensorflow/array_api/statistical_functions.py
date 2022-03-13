# global
import tensorflow as tf
from typing import Union, Tuple, Optional, List
from tensorflow.python.types.core import Tensor


def var(x: Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) -> Tensor:
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    return tf.reduce_mean(tf.square(x - m), axis=axis, keepdims=keepdims)


def min(x: Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> Tensor:
    return tf.math.reduce_min(x, axis = axis, keepdims = keepdims)
