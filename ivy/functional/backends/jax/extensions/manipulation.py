from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp

from ivy.functional.backends.jax import JaxArray


def moveaxis(
    a: JaxArray,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.moveaxis(a, source, destination)


def heaviside(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.heaviside(x1, x2)


def flipud(
    m: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.flipud(m)


def vstack(
    arrays: Sequence[JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.vstack(arrays)


def hstack(
    arrays: Sequence[JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.hstack(arrays)


def rot90(
    m: JaxArray,
    /,
    *,
    k: Optional[int] = 1,
    axes: Optional[Tuple[int, int]] = (0, 1),
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.rot90(m, k, axes)
