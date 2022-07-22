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


argmax.support_native_out = True


def argmin(
    x: JaxArray,
    axis: Optional[int] = None,
    keepdims: bool = False,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.argmin(x, axis=axis, out=out, keepdims=keepdims)


argmin.support_native_out = True


def nonzero(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.nonzero(x)


def where(
    condition: JaxArray, x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.where(condition, x1, x2)
