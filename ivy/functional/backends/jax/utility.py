# global
from typing import Optional, Sequence, Union

import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def all(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x = jnp.array(x, dtype="bool")
    return jnp.all(x, axis, keepdims=keepdims)


def any(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x = jnp.array(x, dtype="bool")
    return jnp.any(x, axis, keepdims=keepdims)
