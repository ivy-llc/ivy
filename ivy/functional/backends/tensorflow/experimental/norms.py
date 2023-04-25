import tensorflow as tf
from typing import Union, Optional, Tuple


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
) -> Tuple[
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
]:
    ndims = len(x.shape)
    runningmean = mean
    runningvariance = variance
    if training:
        n = (
            tf.size(x)
            if ndims == 1
            else tf.cast(tf.divide(tf.size(x), tf.shape(x)[-1]), x.dtype)
        )
        n = tf.cast(n, x.dtype)
        dims = (0, *range(1, ndims - 1))
        mean = tf.math.reduce_mean(x, axis=dims)
        variance = tf.math.reduce_variance(x, axis=dims)
        runningmean = (1 - momentum) * runningmean + momentum * mean
        runningvariance = (1 - momentum) * runningvariance + momentum * variance * n / (
            n - 1
        )
    xnormalized = tf.nn.batch_normalization(x, mean, variance, offset, scale, eps)
    return xnormalized, runningmean, runningvariance


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
    momentum: float = 1e-1,
    out: Optional[tf.Tensor] = None,
) -> Tuple[
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
]:
    # Instance Norm with (N,H,W,C) is the same as BatchNorm with (1, H, W, N*C)
    xdims = len(x.shape)
    N = x.shape[0]
    C = x.shape[-1]
    S = x.shape[1:-1]
    x = tf.transpose(x, perm=(*range(1, xdims - 1), 0, xdims - 1))
    x = tf.reshape(x, (1, *S, N * C))
    mean = tf.tile(mean, [N])
    variance = tf.tile(variance, [N])
    scale = tf.tile(scale, [N])
    offset = tf.tile(offset, [N])
    xnormalized, runningmean, runningvariance = batch_norm(
        x,
        mean,
        variance,
        scale=scale,
        offset=offset,
        training=training,
        eps=eps,
        momentum=momentum,
        out=out,
    )
    xnormalized = tf.reshape(xnormalized, (*S, N, C))
    return (
        tf.transpose(xnormalized, perm=(xdims - 2, *range(0, xdims - 2), xdims - 1)),
        tf.reduce_mean(tf.reshape(runningmean, (N, C)), axis=0),
        tf.reduce_mean(tf.reshape(runningvariance, (N, C)), axis=0),
    )


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
