import tensorflow as tf
from typing import Union, Optional


def l2_normalize(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    denorm = tf.norm(x, axis=axis, keepdims=True)
    denorm = tf.math.maximum(denorm, 1e-12)
    return tf.math.divide(x, denorm)


def batch_norm(
    x: Union[tf.Tensor, tf.Variable],
    mean: Union[tf.Tensor, tf.Variable],
    variance: Union[tf.Tensor, tf.Variable],
    /,
    *,
    scale: Optional[Union[tf.Tensor, tf.Variable]] = None,
    offset: Optional[Union[tf.Tensor, tf.Variable]] = None,
    training: bool = False,
    eps: float = 1e-5,
    momentum: float = 1e-1,
    out: Optional[tf.Tensor] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ndims = len(x.shape)
    runningmean = mean
    runningvariance = variance
    n = tf.cast(tf.divide(tf.size(x), tf.shape(x)[1]), x.dtype)
    if training:
        dims = (0, *range(2, ndims))
        mean = tf.math.reduce_mean(x, axis=dims)
        variance = tf.math.reduce_variance(x, axis=dims)
        runningmean = (1 - momentum) * runningmean + momentum * mean
        runningvariance = (1 - momentum) * runningvariance + momentum * variance * n / (n - 1)
    x = tf.transpose(x, perm=(0, *range(2, ndims), 1))
    ret = tf.nn.batch_normalization(x, mean, variance, offset, scale, eps)
    result = tf.transpose(ret, perm=(0, ndims - 1, *range(1, ndims - 1)))
    return result, runningmean, runningvariance


def instance_norm(
    x: Union[tf.Tensor, tf.Variable],
    mean: Union[tf.Tensor, tf.Variable],
    variance: Union[tf.Tensor, tf.Variable],
    /,
    *,
    scale: Optional[Union[tf.Tensor, tf.Variable]] = None,
    offset: Optional[Union[tf.Tensor, tf.Variable]] = None,
    training: bool = False,
    eps: float = 1e-5,
) -> Union[tf.Tensor, tf.Variable]:
    ndims = len(x.shape)
    if training:
        dims = (*range(2, ndims),)
        mean = tf.math.reduce_mean(x, axis=dims)
        variance = tf.math.reduce_variance(x, axis=dims)
    x = tf.transpose(x, perm=(*range(2, ndims), 1, 0))
    ret = tf.nn.batch_normalization(x, mean, variance, offset, scale, eps)
    return tf.transpose(ret, perm=(ndims - 1, ndims - 2, *range(0, ndims - 2)))


def lp_normalize(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    denorm = tf.norm(x, ord=p, axis=axis, keepdims=True)
    denorm = tf.math.maximum(denorm, 1e-12)
    return tf.math.divide(x, denorm)
