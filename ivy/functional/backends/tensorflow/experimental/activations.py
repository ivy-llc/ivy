from typing import Optional, Union, Literal

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from . import backend_version


def logit(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    eps: Optional[float] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[Tensor] = None,
) -> Tensor:
    x_dtype = x.dtype
    if eps is None:
        x = tf.where(tf.math.logical_or(x > 1, x < 0), ivy.nan, x)
    else:
        x = tf.clip_by_value(x, eps, 1 - eps)
    return tf.cast(tf.math.log(x / (1 - x)), x_dtype)


@with_unsupported_dtypes({"2.15.0 and below": ("complex", "bool")}, backend_version)
def thresholded_relu(
    x: Tensor,
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[Tensor] = None,
) -> Tensor:
    threshold = tf.cast(threshold, x.dtype)
    return tf.cast(tf.where(x > threshold, x, 0), x.dtype)


def relu6(x: Tensor, /, *, complex_mode="jax", out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.relu6(x)


@with_supported_dtypes({"2.15.0 and below": ("float",)}, backend_version)
def logsigmoid(
    input: Tensor, /, *, complex_mode="jax", out: Optional[Tensor] = None
) -> Tensor:
    if input.dtype in [tf.complex64, tf.complex128]:
        return tf.math.log(tf.nn.sigmoid(input))
    return tf.math.log_sigmoid(input)


@with_supported_dtypes({"2.15.0 and below": ("float",)}, backend_version)
def selu(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    ret = tf.nn.selu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_supported_dtypes({"2.15.0 and below": ("float",)}, backend_version)
def silu(
    x: Tensor,
    /,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    ret = tf.nn.silu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_supported_dtypes({"2.15.0 and below": ("float",)}, backend_version)
def elu(x: Tensor, /, *, alpha: float = 1.0, out: Optional[Tensor] = None) -> Tensor:
    ret = tf.keras.activations.elu(x, alpha)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_supported_dtypes({"2.15.0 and below": ("float",)}, backend_version)
def hardtanh(
    x: Tensor,
    /,
    *,
    max_val: float = 1.0,
    min_val: float = -1.0,
    out: Optional[Tensor] = None,
) -> Tensor:
    ret = tf.where(
        tf.math.greater(x, max_val),
        max_val,
        tf.where(tf.math.less(x, min_val), min_val, x),
    )
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_supported_dtypes({"2.15.0 and below": ("float",)}, backend_version)
def tanhshrink(
    x: Tensor,
    /,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    ret = tf.math.subtract(x, tf.math.tanh(x))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_supported_dtypes({"2.15.0 and below": ("float",)}, backend_version)
def threshold(
    x: Tensor,
    /,
    *,
    threshold: Union[int, float],
    value: Union[int, float],
    out: Optional[Tensor] = None,
) -> Tensor:
    ret = tf.where(x > threshold, x, value)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_supported_dtypes({"2.15.0 and below": ("float",)}, backend_version)
def softshrink(
    x: Tensor,
    /,
    *,
    lambd: float = 0.5,
    out: Optional[Tensor] = None,
) -> Tensor:
    ret = tf.where(
        tf.math.greater(x, lambd),
        x - lambd,
        tf.where(tf.math.less(x, -lambd), x + lambd, 0),
    )
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_unsupported_dtypes({"2.15.0 and below": ("complex",)}, backend_version)
def celu(
    x: Tensor,
    /,
    *,
    alpha: float = 1.0,
    complex_mode="jax",
    out: Optional[Tensor] = None,
) -> Tensor:
    return tf.math.maximum(0, x) + alpha * tf.math.expm1(tf.math.minimum(0, x) / alpha)


@with_unsupported_dtypes({"2.15.0 and below": ("uint16",)}, backend_version)
def scaled_tanh(
    x: Tensor,
    /,
    *,
    alpha: float = 1.7159,
    beta: float = 0.67,
    out: Optional[Tensor] = None,
) -> Tensor:
    return alpha * tf.nn.tanh(beta * x)


@with_supported_dtypes({"2.15.0 and below": ("float",)}, backend_version)
def hardshrink(
    x: Tensor,
    /,
    *,
    lambd: float = 0.5,
    out: Optional[Tensor] = None,
) -> Tensor:
    ret = tf.where(
        tf.math.greater(x, lambd),
        x,
        tf.where(tf.math.less(x, -lambd), x, 0),
    )
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


@with_unsupported_dtypes({"2.14.0 and below": ("complex",)}, backend_version)
def hardsilu(
    x: Tensor, /, *, complex_mode="jax", out: Optional[Tensor] = None
) -> Tensor:
    ret = tf.multiply(x, tf.nn.relu6(tf.math.add(x, 3)) / 6)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)
