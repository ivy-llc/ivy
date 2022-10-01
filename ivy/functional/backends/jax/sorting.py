# global
import jax.numpy as jnp
from typing import Optional

# local
import ivy
from ivy.functional.backends.jax import JaxArray

def argsort(
    x: JaxArray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:

    x = -1 * jnp.searchsorted(jnp.unique(x), x) if descending else x
    kind = "stable" if stable else "quicksort"
    return jnp.argsort(x, axis, kind=kind)


def sort(
    x: JaxArray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    kind = "stable" if stable else "quicksort"
    ret = jnp.asarray(jnp.sort(x, axis=axis, kind=kind))
    if descending:
        ret = jnp.asarray(jnp.flip(ret, axis=axis))
    return ret


def searchsorted(
    x: JaxArray,
    v: JaxArray,
    /,
    *,
    side="left",
    sorter=None,
    ret_dtype=jnp.int64,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if ivy.as_native_dtype(ret_dtype) not in [jnp.int32, jnp.int64]:
        raise ValueError("only int32 and int64 are supported for ret_dtype.")
    return jnp.searchsorted(x, v, sorter=sorter, side=side).astype(ret_dtype)
