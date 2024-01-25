# global
import jax.numpy as jnp
from typing import Union, Optional, Sequence

# local
from ivy.functional.backends.jax import JaxArray
import ivy


def all(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x = jnp.array(x, dtype="bool")
    try:
        return jnp.all(x, axis, keepdims=keepdims)
    except ValueError as error:
        raise ivy.utils.exceptions.IvyIndexError(error) from error


def any(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x = jnp.array(x, dtype="bool")
    try:
        return jnp.any(x, axis, keepdims=keepdims, out=out)
    except ValueError as error:
        raise ivy.utils.exceptions.IvyIndexError(error) from error
