import tensorflow as tf
from typing import Literal, Union, Optional, Tuple
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
from . import backend_version
import math


@with_unsupported_dtypes({"2.15.0 and below": "uint8"}, backend_version)
def l1_normalize(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    denorm = tf.norm(x, ord=1, axis=axis, keepdims=True)
    denorm = tf.math.maximum(denorm, 1e-12)
    return tf.math.divide(x, denorm)


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


@with_supported_dtypes({"2.15.0 and below": ("float32", "float16")}, backend_version)
def local_response_norm(
    x: Union[tf.Tensor, tf.Variable],
    size,
    /,
    *,
    bias: Optional[float] = 1.0,
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.5,
    average: bool = False,
    data_format: Optional[Literal["NHWC", "NCHW"]] = "NHWC",
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    # `alpha = alpha/size if average else alpha` was causing numerical instability
    if average:
        ret = tf.nn.local_response_normalization(
            x / math.sqrt(size),
            depth_radius=size // 2,
            bias=bias,
            alpha=alpha,
            beta=beta,
        ) * math.sqrt(size)
    else:
        ret = tf.nn.local_response_normalization(
            x, depth_radius=size // 2, bias=bias, alpha=alpha, beta=beta
        )
    if data_format == "NCHW":
        ret = tf.transpose(ret, (0, 3, 1, 2))
    return ret


local_response_norm.partial_mixed_handler = lambda x, size, **kwargs: size % 2 != 0


@with_unsupported_dtypes({"2.15.0 and below": ("float16", "bfloat16")}, backend_version)
def batch_norm(
    x: Union[tf.Tensor, tf.Variable],
    mean: Optional[Union[tf.Tensor, tf.Variable]],
    variance: Optional[Union[tf.Tensor, tf.Variable]],
    /,
    *,
    scale: Optional[Union[tf.Tensor, tf.Variable]] = None,
    offset: Optional[Union[tf.Tensor, tf.Variable]] = None,
    training: Optional[bool] = False,
    eps: Optional[float] = 1e-5,
    momentum: Optional[float] = 1e-1,
    data_format: Optional[str] = "NSC",
    out: Optional[
        Tuple[
            Union[tf.Tensor, tf.Variable],
            Union[tf.Tensor, tf.Variable],
            Union[tf.Tensor, tf.Variable],
        ]
    ] = None,
) -> Tuple[
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
]:
    xdims = len(x.shape)
    if data_format == "NCS":
        x = tf.transpose(x, perm=(0, *range(2, xdims), 1))

    runningmean = mean
    runningvariance = variance
    if training:
        n = tf.size(x) if xdims == 1 else tf.divide(tf.size(x), tf.shape(x)[-1])
        n = tf.cast(n, x.dtype) if n.dtype != x.dtype else n
        dims = (0, *range(1, xdims - 1))
        mean = tf.math.reduce_mean(x, axis=dims)
        variance = tf.math.reduce_variance(x, axis=dims)
        runningmean = (
            ((1 - momentum) * runningmean + momentum * mean)
            if runningmean is not None
            else runningmean
        )
        runningvariance = (
            (1 - momentum) * runningvariance + momentum * variance * n / (n - 1)
            if runningvariance is not None
            else runningvariance
        )

    inv = 1.0 / tf.math.sqrt(variance + eps)
    offset = 0 if offset is None else offset
    if scale is not None:
        inv = tf.math.multiply(inv, scale)
    xnormalized = tf.math.add(tf.math.multiply(x, inv), offset)
    xnormalized = tf.math.subtract(xnormalized, tf.math.multiply(mean, inv))
    # the above approach is faster than tf.nn.batch_normalization

    if data_format == "NCS":
        xnormalized = tf.transpose(
            xnormalized, perm=(0, xdims - 1, *range(1, xdims - 1))
        )

    return xnormalized, runningmean, runningvariance


def instance_norm(
    x: Union[tf.Tensor, tf.Variable],
    mean: Optional[Union[tf.Tensor, tf.Variable]] = None,
    variance: Optional[Union[tf.Tensor, tf.Variable]] = None,
    /,
    *,
    scale: Optional[Union[tf.Tensor, tf.Variable]] = None,
    offset: Optional[Union[tf.Tensor, tf.Variable]] = None,
    training: Optional[bool] = False,
    eps: Optional[float] = 1e-5,
    momentum: Optional[float] = 1e-1,
    data_format: Optional[str] = "NSC",
    out: Optional[
        Tuple[
            Union[tf.Tensor, tf.Variable],
            Union[tf.Tensor, tf.Variable],
            Union[tf.Tensor, tf.Variable],
        ]
    ] = None,
) -> Tuple[
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
]:
    # Instance Norm with (N,H,W,C) is the same as BatchNorm with (1, H, W, N*C)
    xdims = len(x.shape)
    if data_format == "NCS":
        x = tf.transpose(x, perm=(*range(2, xdims), 0, 1))
    elif data_format == "NSC":
        x = tf.transpose(x, perm=(*range(1, xdims - 1), 0, xdims - 1))
    else:
        raise ValueError(f"Invalid data_format: {data_format}.")

    N = x.shape[-2]
    C = x.shape[-1]
    S = x.shape[0:-2]
    x = tf.reshape(x, (1, *S, N * C))
    mean = tf.tile(mean, [N]) if mean is not None else mean
    variance = tf.tile(variance, [N]) if variance is not None else variance
    if scale is not None:
        scale = tf.tile(scale, [N])
    if offset is not None:
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
    )
    xnormalized = tf.reshape(xnormalized, (*S, N, C))
    if data_format == "NCS":
        xnormalized = tf.transpose(
            xnormalized, perm=(xdims - 2, xdims - 1, *range(0, xdims - 2))
        )
    else:
        xnormalized = tf.transpose(
            xnormalized, perm=(xdims - 2, *range(0, xdims - 2), xdims - 1)
        )

    runningmean = (
        tf.reduce_mean(tf.reshape(runningmean, (N, C)), axis=0)
        if runningmean is not None
        else runningmean
    )
    runningvariance = (
        tf.reduce_mean(tf.reshape(runningvariance, (N, C)), axis=0)
        if runningvariance is not None
        else runningvariance
    )

    return (
        xnormalized,
        runningmean,
        runningvariance,
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
