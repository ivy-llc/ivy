from typing import Optional, Tuple

import ivy
import jax.numpy as jnp

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
) -> Tuple[JaxArray]:
    return jnp.nonzero(x)


def where(
    condition: JaxArray,
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.where(condition, x1, x2)


# Extra #
# ----- #


def argwhere(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.argwhere(x)
