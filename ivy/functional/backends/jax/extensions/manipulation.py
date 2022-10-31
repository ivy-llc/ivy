# local
from typing import Optional, Union, Sequence, Tuple, NamedTuple
import jax.numpy as jnp

# local
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


def top_k(
    x: JaxArray,
    k: int,
    /,
    *,
    axis: Optional[int] = -1,
    largest: Optional[bool] = True,
    out: Optional[Tuple[JaxArray, JaxArray]] = None,
) -> Tuple[JaxArray, JaxArray]:
    if not largest:
        indices = jnp.argsort(x, axis=axis)
        indices = jnp.take(indices, jnp.arange(k), axis=axis)
    else:
        x *= -1
        indices = jnp.argsort(x, axis=axis)
        indices = jnp.take(indices, jnp.arange(k), axis=axis)
        x *= -1
    topk_res = NamedTuple("top_k", [("values", JaxArray), ("indices", JaxArray)])
    val = jnp.take_along_axis(x, indices, axis=axis)
    return topk_res(val, indices)
