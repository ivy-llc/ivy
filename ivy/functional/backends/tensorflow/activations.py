"""
TensorFlow activation functions.

Collection of TensorFlow activation functions, wrapped to fit Ivy syntax
and signature.
"""

from typing import Optional, Union

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def gelu(
    x: Tensor, /, *, approximate: bool = False, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.gelu(x, approximate)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def selu(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.selu(x)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def leaky_relu(
    x: Tensor, /, *, alpha: float = 0.2, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.leaky_relu(x, alpha)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def relu(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.relu(x)


def sigmoid(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    if not ivy.is_array(x):
        x = float(x)
    return tf.nn.sigmoid(x)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def softmax(
    x: Tensor, /, *, axis: Optional[int] = None, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.softmax(x, axis)


@with_supported_dtypes(
    {"2.13.0 and below": ("float16", "bfloat16", "float32", "float64")}, backend_version
)
def softplus(
    x: Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    if beta is not None and beta != 1:
        x_beta = x * beta
        res = (tf.nn.softplus(x_beta)) / beta
    else:
        x_beta = x
        res = tf.nn.softplus(x)
    if threshold is not None:
        return tf.where(x_beta > threshold, x, res)
    return res


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def log_softmax(
    x: Tensor, /, *, axis: Optional[int] = None, out: Optional[Tensor] = None
):
    return tf.nn.log_softmax(x, axis)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def mish(
    x: Tensor,
    /,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    return x * tf.math.tanh(tf.math.softplus(x))


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def hardswish(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return x * tf.nn.relu6(x + 3) / 6


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def softsign(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.softsign(x)
