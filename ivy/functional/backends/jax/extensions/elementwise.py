from typing import Optional
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp


def lcm(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.lcm(x1, x2)


def sinc(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinc(x)


def fmod(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fmod(x1, x2)


def fmax(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fmax(x1, x2)


def trapz(
    y: JaxArray,
    /,
    *,
    x: Optional[JaxArray] = None,
    dx: Optional[float] = 1.0,
    axis: Optional[int] = - 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.trapz(y, x=x, dx=dx, axis=axis)
