# global
import jax.numpy as jnp
from typing import Optional

# local
from ivy.functional.backends.jax import JaxArray


def argsort(
    x: JaxArray,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if descending:
        ret = jnp.asarray(
            jnp.argsort(-1 * jnp.searchsorted(jnp.unique(x), x), axis, kind="stable")
        )
    else:
        ret = jnp.asarray(jnp.argsort(x, axis, kind="stable"))
    return ret


def sort(
    x: JaxArray,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    kind = "stable" if stable else "quicksort"
    res = jnp.asarray(jnp.sort(x, axis=axis, kind=kind))
    if descending:
        ret = jnp.asarray(jnp.flip(res, axis=axis))
    else:
        ret = res
    return ret


def searchsorted(
    x: JaxArray,
    v: JaxArray,
    side="left",
    sorter=None,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    res = jnp.asarray(jnp.searchsorted(x, v, side=side))
    return res
