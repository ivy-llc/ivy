# global
import tensorflow as tf
from typing import Union, Tuple, Optional, List
from tensorflow.python.types.core import Tensor

# local
import ivy


# noinspection PyShadowingBuiltins
def all(
    x: Tensor,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False
) -> Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = tf.reduce_all(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
    return ret


# noinspection PyShadowingBuiltins
def any(
    x: Tensor,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False
) -> Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = tf.reduce_any(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
    return ret
