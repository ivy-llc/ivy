# global
import tensorflow as tf
from typing import Union, Tuple, Optional, List


def all(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False,
) -> Union[tf.Tensor, tf.Variable]:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return tf.reduce_all(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)


def any(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False,
) -> Union[tf.Tensor, tf.Variable]:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return tf.reduce_any(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
