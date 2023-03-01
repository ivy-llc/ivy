from typing import Optional, Union

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def logit(
    x: Union[tf.Tensor, tf.Variable], /, *, eps: Optional[float] = None, out=None
):
    x_dtype = x.dtype
    if eps is None:
        x = tf.where(tf.math.logical_or(x > 1, x < 0), ivy.nan, x)
    else:
        x = tf.clip_by_value(x, eps, 1 - eps)
    return tf.cast(tf.math.log(x / (1 - x)), x_dtype)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def thresholded_relu(
    x: Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = 0,
    out: Optional[Tensor] = None,
) -> Tensor:
    return tf.where(x > threshold, x, 0)


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
    return tf.transpose(ret, perm=(0, ndims-1, *range(1, ndims-1)))
