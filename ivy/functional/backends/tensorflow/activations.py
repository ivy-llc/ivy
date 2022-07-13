"""Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy


def relu(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    ret = tf.nn.relu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def leaky_relu(x: Tensor,
    alpha: Optional[float] = 0.2,
    out: Optional[Tensor] = None
) -> Tensor:
    ret = tf.nn.leaky_relu(x, alpha)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def gelu(x: Tensor,
    approximate: Optional[bool] = True,
    out: Optional[Tensor] = None
) -> Tensor:
    ret = tf.nn.gelu(x, approximate)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sigmoid(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    ret = tf.nn.sigmoid(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def tanh(x: Tensor,out: Optional[Tensor] = None) -> Tensor:
    ret = tf.nn.tanh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def softmax(x: Tensor, axis: Optional[int] = None,out: Optional[Tensor] = None) -> Tensor:
    ret = tf.exp(x) / tf.reduce_sum(tf.exp(x), axis, keepdims=True)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def softplus(x: Tensor,out: Optional[Tensor] = None) -> Tensor:
    ret = tf.nn.softplus(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret

