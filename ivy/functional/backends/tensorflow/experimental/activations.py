from typing import Optional, Union

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
    out: Optional[Tensor] = None,
) -> Tensor:
    x_dtype = x.dtype
    if eps is None:
        x = tf.where(tf.math.logical_or(x > 1, x < 0), ivy.nan, x)
    else:
        x = tf.clip_by_value(x, eps, 1 - eps)
    return tf.cast(tf.math.log(x / (1 - x)), x_dtype)


@with_unsupported_dtypes({"2.9.1 and below": ("complex", "bool")}, backend_version)
def thresholded_relu(
    x: Tensor,
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[Tensor] = None,
) -> Tensor:
    x, threshold = ivy.promote_types_of_inputs(x, threshold)
    return tf.cast(tf.where(x > threshold, x, 0), x.dtype)


def hardshrink(
    x: Tensor,
    /,
    *,
    lambd: Optional[float] = 0.5,
    out: Optional[Tensor] = None,
):
    mask = tf.logical_or(tf.greater(x, lambd), tf.less(x, -lambd))
    return tf.where(mask, x, 0.0)


def softshrink(
    x: Tensor,
    /,
    *,
    lambd: Optional[float] = 0.5,
    out: Optional[Tensor] = None,
):
    low = tf.where(tf.less(x, -lambd), tf.add(x, lambd), 0)
    up = tf.where(tf.greater(x, lambd), tf.subtract(x, lambd), 0)
    return tf.add(low, up)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def threshold(
    x: Tensor,
    threshold: Union[int, float],
    value: Union[int, float],
    /,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    return tf.where(x > threshold, x, value)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def relu6(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.relu6(x)


def batch_norm(
    x: Tensor,
    mean: Tensor,
    variance: Tensor,
    /,
    *,
    scale: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    training: bool = False,
    eps: float = 1e-5,
):
    ndims = len(x.shape)
    if training:
        dims = (0, *range(2, ndims))
        mean = tf.math.reduce_mean(x, axis=dims)
        variance = tf.math.reduce_variance(x, axis=dims)
    x = tf.transpose(x, perm=(0, *range(2, ndims), 1))
    ret = tf.nn.batch_normalization(x, mean, variance, offset, scale, eps)
    return tf.transpose(ret, perm=(0, ndims - 1, *range(1, ndims - 1)))


def logsigmoid(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.math.log_sigmoid(x)


@with_unsupported_dtypes({"2.9.1 and below": ("complex", "bfloat16")}, backend_version)
def selu(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.selu(x)


def sigmoid(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    if not ivy.is_array(x):
        x = float(x)
    return tf.nn.sigmoid(x)


def hard_sigmoid(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    if not ivy.is_array(x):
        x = float(x)
    return relu6(x + 3.0) / 6


def hard_silu(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    if not ivy.is_array(x):
        x = float(x)
    return x * hard_sigmoid(x)


def hard_tanh(
    x: Tensor, /, *, min_value=-1.0, max_value=1.0, out: Optional[Tensor] = None
) -> Tensor:
    return tf.clip_by_value(x, clip_value_min=min_value, clip_value_max=max_value)


def softsign(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.softsign(x)


def silu(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.silu(x)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def elu(x: Tensor, /, *, alpha: float = 1.0, out: Optional[Tensor] = None) -> Tensor:
    return tf.keras.activations.elu(x, alpha=alpha)


def parametric_relu(
    x: Tensor, weight: Union[float, Tensor], /, *, out: Optional[Tensor] = None
) -> Tensor:
    return tf.where(x >= 0, x, weight * x)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def celu(x: Tensor, /, *, alpha: float = 1.0, out: Optional[Tensor] = None) -> Tensor:
    return tf.where(x > 0, x, alpha * tf.subtract(tf.exp(x), 1.0))


def glu(x: Tensor, /, *, axis: int = -1, out: Optional[Tensor] = None) -> Tensor:
    assert x.shape[axis] % 2 == 0, "axis size must be divisible by 2"
    x1, x2 = tf.split(x, 2, axis)
    return x1 * tf.sigmoid(x2)
