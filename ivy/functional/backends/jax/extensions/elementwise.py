from typing import Optional, Union
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
    axis: Optional[int] = -1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.trapz(y, x=x, dx=dx, axis=axis)


def float_power(
    x1: Union[JaxArray, float, list, tuple],
    x2: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.float_power(x1, x2)


def exp2(
    x: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.exp2(x)
