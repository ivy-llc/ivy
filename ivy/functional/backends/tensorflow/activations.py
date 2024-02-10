"""TensorFlow activation functions.

Collection of TensorFlow activation functions, wrapped to fit Ivy syntax
and signature.
"""

from typing import Optional, Union, Literal

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from . import backend_version
import ivy.functional.backends.tensorflow as tf_backend


def gelu(
    x: Tensor,
    /,
    *,
    approximate: bool = False,
    complex_mode="jax",
    out: Optional[Tensor] = None,
) -> Tensor:
    if x.dtype in [tf.complex64, tf.complex128]:
        return 0.5 * x * (1 + tf.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    return tf.nn.gelu(x, approximate)


def leaky_relu(
    x: Tensor,
    /,
    *,
    alpha: float = 0.2,
    complex_mode="jax",
    out: Optional[Tensor] = None,
) -> Tensor:
    return tf.nn.leaky_relu(x, alpha)


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "float",
            "int",
            "complex",
        )
    },
    backend_version,
)
def relu(x: Tensor, /, *, complex_mode="jax", out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.relu(x)


def sigmoid(
    x: Tensor, /, *, complex_mode="jax", out: Optional[Tensor] = None
) -> Tensor:
    return 1 / (1 + tf.exp(-x))


def softmax(
    x: Tensor, /, *, axis: Optional[int] = None, out: Optional[Tensor] = None
) -> Tensor:
    if axis is None:
        axis = -1
    dtype = x.dtype
    if "complex" in str(dtype):
        amax = tf_backend.max(x, axis=axis, keepdims=True)
        normalized = tf.exp(tf.subtract(x, amax))
        return tf.divide(
            normalized, tf.reduce_sum(normalized, axis=axis, keepdims=True)
        )
    return tf.nn.softmax(x, axis)


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    backend_version,
)
def softplus(
    x: Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    complex_mode="jax",
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


# Softsign
@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    backend_version,
)
def softsign(x: tf.Tensor, /, out: Optional[tf.Tensor] = None) -> tf.Tensor:
    return tf.nn.softsign(x)


def log_softmax(
    x: Tensor,
    /,
    *,
    axis: Optional[int] = -1,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[Tensor] = None,
):
    if "complex" in str(x.dtype):
        x_max = tf_backend.max(x, axis=axis, keepdims=True)
        sub_temp = tf.subtract(x, x_max)
        ret = tf.reduce_sum(tf.exp(sub_temp), axis=axis, keepdims=True)
        ret = tf.math.log(ret)
        return tf.subtract(sub_temp, ret)
    return tf.nn.log_softmax(x, axis)


def mish(
    x: Tensor,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[Tensor] = None,
) -> Tensor:
    if "complex" in str(x.dtype):
        x_norm = tf.math.log1p(tf.exp(x))
    else:
        x_norm = tf.math.softplus(x)
    return tf.multiply(x, tf.math.tanh(x_norm))


@with_unsupported_dtypes({"2.15.0 and below": ("complex",)}, backend_version)
def hardswish(
    x: Tensor,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[Tensor] = None,
) -> Tensor:
    return x * tf.nn.relu6(x + 3) / 6
