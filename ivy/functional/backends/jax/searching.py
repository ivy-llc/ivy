from typing import Optional, Tuple

import jax.numpy as jnp

import ivy
from ivy.functional.backends.jax import JaxArray


# Array API Standard #
# ------------------ #


def argmax(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.argmax(x, axis=axis, keepdims=keepdims)


def argmin(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.argmin(x, axis=axis, keepdims=keepdims)


def nonzero(
    x: JaxArray,
    /,
    *,
    as_tuple: bool = True,
) -> Tuple[JaxArray]:
    if as_tuple:
        return jnp.nonzero(x)
    else:
        return jnp.stack(jnp.nonzero(x), axis=1)


def where(
    condition: JaxArray,
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.where(condition, x1, x2).astype(x1.dtype)


# Extra #
# ----- #


def argwhere(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.argwhere(x)
