import jax.numpy as jnp

from ivy.functional.backends.jax import JaxArray
from typing import Optional


def argmax(
    x: JaxArray,
    axis: Optional[int] = None,
    keepdims: bool = False,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.argmax(x, axis=axis, out=out, keepdims=keepdims)


def argmin(
    x: JaxArray,
    axis: Optional[int] = None,
    keepdims: bool = False,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.argmin(x, axis=axis, out=out, keepdims=keepdims)


def nonzero(x: JaxArray) -> JaxArray:
    return jnp.nonzero(x)


def where(condition: JaxArray, x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.where(condition, x1, x2)
