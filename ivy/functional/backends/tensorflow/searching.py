# global
from typing import Optional, Union, Tuple

import ivy
import tensorflow as tf


# Array API Standard #
# ------------------ #


def argmax(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = x.numpy().argmax(axis=axis, keepdims=keepdims)
    return tf.convert_to_tensor(ret, dtype=ret.dtype)


def argmin(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = x.numpy().argmin(axis=axis, keepdims=keepdims)
    return tf.convert_to_tensor(ret, dtype=ret.dtype)


def nonzero(
    x: Union[tf.Tensor, tf.Variable],
    /,
) -> Tuple[Union[tf.Tensor, tf.Variable]]:
    return tuple(tf.experimental.numpy.nonzero(x))


def where(
    condition: Union[tf.Tensor, tf.Variable],
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.cast(tf.experimental.numpy.where(condition, x1, x2), x1.dtype)


# Extra #
# ----- #


def argwhere(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    where_x = tf.experimental.numpy.where(x)
    if len(where_x) == 1:
        return tf.expand_dims(where_x[0], -1)
    res = tf.experimental.numpy.concatenate(
        [tf.expand_dims(item, -1) for item in where_x], -1
    )
    return res
