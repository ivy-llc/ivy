# global
import tensorflow as tf
from typing import Optional
from tensorflow.python.types.core import Tensor


def argmax(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:

    ret = tf.constant(x).numpy().argmax(axis=axis, keepdims=keepdims)
    ret_dtype = ret.dtype
    ret = tf.convert_to_tensor(ret, dtype=ret_dtype)

    return ret


def argmin(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    ret = x.numpy().argmin(axis=axis, keepdims=keepdims)
    ret = tf.convert_to_tensor(ret, dtype=ret.dtype)
    return ret


def nonzero(x: Tensor) -> Tensor:
    return tf.experimental.numpy.nonzero(x)


def where(condition: Tensor, x1: Tensor, x2: Tensor) -> Tensor:
    return tf.experimental.numpy.where(condition, x1, x2)
