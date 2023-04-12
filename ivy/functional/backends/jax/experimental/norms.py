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
