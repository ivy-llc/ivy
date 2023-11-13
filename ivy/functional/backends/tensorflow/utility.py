# global
import tensorflow as tf
from typing import Union, Optional, Sequence


# local
import ivy


def all(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x = tf.constant(x)
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    try:
        return tf.reduce_all(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
    except tf.errors.InvalidArgumentError as e:
        raise ivy.utils.exceptions.IvyIndexError(e)


def any(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x = tf.constant(x)
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    try:
        return tf.reduce_any(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
    except tf.errors.InvalidArgumentError as e:
        raise ivy.utils.exceptions.IvyIndexError(e)
