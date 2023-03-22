import jax.numpy as jnp
from typing import Optional
from ivy.functional.backends.jax import JaxArray
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes({"0.3.14 and below": ("float16",)}, backend_version)
def l2_normalize(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if axis is None:
        denorm = jnp.linalg.norm(x.flatten(), 2, axis)
    else:
        denorm = jnp.linalg.norm(x, 2, axis, keepdims=True)
    denorm = jnp.maximum(denorm, 1e-12)
    return x / denorm


@with_unsupported_dtypes({"0.3.14 and below": ("float16", "bfloat16")}, backend_version)
def batch_norm(
    x: JaxArray,
    mean: JaxArray,
    variance: JaxArray,
    /,
    *,
    scale: Optional[JaxArray] = None,
    offset: Optional[JaxArray] = None,
    training: bool = False,
    eps: float = 1e-5,
    momentum: float = 1e-1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    runningmean = mean
    runningvariance = variance
    n = x.size / x.shape[1]
    ndims = len(x.shape)
    if training:
        dims = (0, *range(2, ndims))
        mean = jnp.mean(x, axis=dims)
        variance = jnp.var(x, axis=dims)
        runningmean = (1 - momentum) * runningmean + momentum * mean
        runningvariance = (1 - momentum) * runningvariance + momentum * variance * n / (n - 1)
    x = jnp.transpose(x, (0, *range(2, ndims), 1))
    inv = 1.0 / jnp.sqrt(variance + eps)
    if scale is not None:
        inv *= scale

    ret = x * inv.astype(x.dtype) + (
        offset - mean * inv if offset is not None else -mean * inv
    ).astype(x.dtype)

    result = jnp.transpose(ret, (0, ndims - 1, *range(1, ndims - 1)))
    return result, runningmean, runningvariance


@with_unsupported_dtypes({"0.3.14 and below": ("float16", "bfloat16")}, backend_version)
def instance_norm(
    x: JaxArray,
    mean: JaxArray,
    variance: JaxArray,
    /,
    *,
    scale: Optional[JaxArray] = None,
    offset: Optional[JaxArray] = None,
    training: bool = False,
    eps: float = 1e-5,
) -> JaxArray:
    ndims = len(x.shape)
    if training:
        dims = (*range(2, ndims),)
        mean = jnp.mean(x, axis=dims)
        variance = jnp.var(x, axis=dims)
    x = jnp.transpose(x, (*range(2, ndims), 1, 0))
    inv = 1.0 / jnp.sqrt(variance + eps)
    if scale is not None:
        inv *= scale

    ret = x * inv.astype(x.dtype) + (
        offset - mean * inv if offset is not None else -mean * inv
    ).astype(x.dtype)

    return jnp.transpose(ret, (ndims - 1, ndims - 2, *range(0, ndims - 2)))


@with_unsupported_dtypes({"0.3.14 and below": ("float16",)}, backend_version)
def lp_normalize(
    x: JaxArray,
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if axis is None:
        denorm = jnp.linalg.norm(x.flatten(), axis=axis, ord=p)
    else:
        denorm = jnp.linalg.norm(x, axis=axis, ord=p, keepdims=True)

    denorm = jnp.maximum(denorm, 1e-12)
    return jnp.divide(x, denorm)
