# global
import tensorflow as tf
from typing import Optional, Union, Tuple


def argmax(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[int] = None,
    keepdims: bool = False,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.constant(x).numpy().argmax(axis=axis, keepdims=keepdims)
    ret = tf.convert_to_tensor(ret, dtype=ret.dtype)

    return ret


def argmin(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[int] = None,
    keepdims: bool = False,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    ret = x.numpy().argmin(axis=axis, keepdims=keepdims)
    ret = tf.convert_to_tensor(ret, dtype=ret.dtype)
    return ret


def nonzero(x: Union[tf.Tensor, tf.Variable]) -> Tuple[Union[tf.Tensor, tf.Variable]]:
    return tuple(tf.experimental.numpy.nonzero(x))


def where(
    condition: Union[tf.Tensor, tf.Variable],
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.where(condition, x1, x2)
