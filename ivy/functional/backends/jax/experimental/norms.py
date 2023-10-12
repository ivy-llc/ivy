import jax.numpy as jnp
from typing import Optional
from ivy.functional.backends.jax import JaxArray


def l1_normalize(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if not isinstance(x, JaxArray):
        x = jnp.array(x)
    if axis is None:
        norm = jnp.sum(jnp.abs(jnp.ravel(x)))
        denorm = norm * jnp.ones_like(x)
    else:
        norm = jnp.sum(jnp.abs(x), axis=axis, keepdims=True)
        denorm = jnp.divide(norm, jnp.abs(x) + 1e-12)
    return jnp.divide(x, denorm)


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
