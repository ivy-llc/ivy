import tensorflow as tf
from typing import Union, Optional


def l2_normalize(
    x: Union[tf.Tensor, tf.Variable], /, *, axis: int = None, out=None
) -> tf.Tensor:

    denorm = tf.norm(x, axis=axis, keepdims=True)
    denorm = tf.math.maximum(denorm, 1e-12)
    return tf.math.divide(x, denorm)


def instance_norm(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    scale: Optional[Union[tf.Tensor, tf.Variable]],
    bias: Optional[Union[tf.Tensor, tf.Variable]],
    eps: float = 1e-05,
    momentum: Optional[float] = 0.1,
    data_format: str = "NCHW",
    running_mean: Optional[Union[tf.Tensor, tf.Variable]] = None,
    running_stddev: Optional[Union[tf.Tensor, tf.Variable]] = None,
    affine: Optional[bool] = True,
    track_running_stats: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    if scale is not None:
        scale = tf.reshape(scale, shape=(1, 1, 1, -1))
    if bias is not None:
        bias = tf.reshape(bias, shape=(1, 1, 1, -1))
    if running_mean is not None:
        running_mean = tf.reshape(running_mean, shape=(1, 1, 1, -1))
    if running_stddev is not None:
        running_stddev = tf.reshape(running_stddev, shape=(1, 1, 1, -1))
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    elif data_format != "NHWC":
        raise NotImplementedError

    mean, var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
    if scale is None:
        scale = tf.ones_like(var)
    if bias is None:
        bias = tf.zeros_like(mean)

    if affine:
        normalized = tf.nn.batch_normalization(x, mean, var, bias, scale, eps)
    else:
        scale_ = tf.ones_like(var)
        bias_ = tf.zeros_like(mean)
        normalized = tf.nn.batch_normalization(x, mean, var, bias_, scale_, eps)
    if track_running_stats:
        if running_mean is None:
            running_mean = tf.zeros_like(mean)
        if running_stddev is None:
            running_stddev = tf.ones_like(var)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_stddev = momentum * running_stddev + (1 - momentum) * tf.sqrt(var)
        if data_format == "NCHW":
            normalized = tf.transpose(normalized, (0, 3, 1, 2))
            running_mean = tf.transpose(running_mean, (0, 3, 1, 2))
            running_stddev = tf.transpose(running_stddev, (0, 3, 1, 2))
        return normalized, running_mean, running_stddev
    if data_format == "NCHW":
        normalized = tf.transpose(normalized, (0, 3, 1, 2))
    return normalized

def lp_normalize(x: Union[tf.Tensor, tf.Variable], /, *, p:float = 2, axis: int = None, out=None) -> tf.Tensor:
    denorm = tf.norm(x, ord=p, axis=axis, keepdims=True)
    denorm = tf.math.maximum(denorm, 1e-12)
    return tf.math.divide(x, denorm)