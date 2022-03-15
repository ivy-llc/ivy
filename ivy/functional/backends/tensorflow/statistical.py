# global
_round = round
import tensorflow as tf
from typing import Tuple, Union


def min(x: tf.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> tf.Tensor:
    return tf.math.reduce_min(x, axis = axis, keepdims = keepdims)

def max(x: tf.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> tf.Tensor:
    return tf.math.reduce_max(x, axis = axis, keepdims = keepdims)