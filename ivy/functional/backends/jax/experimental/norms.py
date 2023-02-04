import jax.numpy as jnp
from typing import Optional
from ivy.functional.backends.jax import JaxArray
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes({"0.3.14 and below": ("float16",)}, backend_version)
def l2_normalize(x: JaxArray, /, *, axis: int = None, out=None) -> JaxArray:
    denorm = jnp.linalg.norm(x, axis=axis, ord=2, keepdims=True)
    denorm = jnp.maximum(denorm, 1e-12)
    return x / denorm


@with_unsupported_dtypes({"0.3.14 and below": ("float16", "bfloat16")}, backend_version)
def instance_norm(
    x: JaxArray,
    /,
    *,
    scale: Optional[JaxArray] = None,
    bias: Optional[JaxArray] = None,
    eps: float = 1e-05,
    momentum: Optional[float] = 0.1,
    data_format: str = "NCHW",
    running_mean: Optional[JaxArray] = None,
    running_stddev: Optional[JaxArray] = None,
    affine: Optional[bool] = True,
    track_running_stats: Optional[bool] = False,
    out: Optional[JaxArray] = None,
):
    if scale is not None:
        scale = jnp.expand_dims(scale, axis=(0, 2, 3))
    if bias is not None:
        bias = jnp.expand_dims(bias, axis=(0, 2, 3))
    if running_mean is not None:
        running_mean = jnp.expand_dims(running_mean, axis=(0, 2, 3))
    if running_stddev is not None:
        running_stddev = jnp.expand_dims(running_stddev, axis=(0, 2, 3))
    if data_format == "NHWC":
        x = jnp.transpose(x, (0, 3, 1, 2))
    elif data_format != "NCHW":
        raise NotImplementedError
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    var = jnp.var(x, axis=(0, 2, 3), keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    if affine:
        if scale is None:
            scale = jnp.ones_like(var)
        if bias is None:
            bias = jnp.zeros_like(mean)
        normalized = scale * normalized + bias
    if track_running_stats:
        if running_mean is None:
            running_mean = jnp.zeros_like(mean)
        if running_stddev is None:
            running_stddev = jnp.ones_like(var)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_stddev = momentum * running_stddev + (1 - momentum) * jnp.sqrt(var)
        if data_format == "NHWC":
            normalized = jnp.transpose(normalized, (0, 2, 3, 1))
            running_mean = jnp.transpose(running_mean, (0, 2, 3, 1))
            running_stddev = jnp.transpose(running_stddev, (0, 2, 3, 1))
        return normalized, running_mean, running_stddev
    if data_format == "NHWC":
        normalized = jnp.transpose(normalized, (0, 2, 3, 1))
    return normalized
