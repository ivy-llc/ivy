# global
import tensorflow as tf
from typing import Union, Tuple, Optional
from tensorflow.python.types.core import Tensor


# noinspection PyShadowingBuiltins
def all(x: Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False)\
        -> Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return tf.reduce_all(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
