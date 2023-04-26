# global
import jax.numpy as jnp
from typing import Optional, Literal, Union, List

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
    side: Literal["left", "right"] = "left",
    sorter: Optional[Union[JaxArray, List[int]]] = None,
    ret_dtype: jnp.dtype = jnp.int64,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    assert ivy.is_int_dtype(ret_dtype), ValueError(
        "only Integer data types are supported for ret_dtype."
    )
    if sorter is not None:
        assert ivy.is_int_dtype(sorter.dtype) and not ivy.is_uint_dtype(
            sorter.dtype
        ), TypeError(
            f"Only signed integer data type for sorter is allowed, got {sorter.dtype}."
        )
        x = jnp.take_along_axis(x, sorter, axis=-1)
    if x.ndim != 1:
        assert x.shape[:-1] == v.shape[:-1], RuntimeError(
            f"the first N-1 dimensions of x array and v array "
            f"must match, got {x.shape} and {v.shape}"
        )
        original_shape = v.shape
        out_array = []  # JAX arrays are immutable.
        x = x.reshape(-1, x.shape[-1])
        v = v.reshape(-1, v.shape[-1])
        for i in range(x.shape[0]):
            out_array.append(jnp.searchsorted(x[i], v[i], side=side))
        ret = jnp.array(out_array).reshape(original_shape)
    else:
        ret = jnp.searchsorted(x, v, side=side)
    return ret.astype(ret_dtype)
