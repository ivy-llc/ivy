# global
import jax.numpy as jnp
from typing import Optional, Literal, Union, List

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.jax import JaxArray
from . import backend_version


def argsort(
    x: JaxArray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    kind = "stable" if stable else "quicksort"
    return (
        jnp.argsort(-x, axis=axis, kind=kind)
        if descending
        else jnp.argsort(x, axis=axis, kind=kind)
    )


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
            "the first N-1 dimensions of x array and v array "
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


# msort
@with_unsupported_dtypes({"0.4.14 and below": ("complex",)}, backend_version)
def msort(
    a: Union[JaxArray,  list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    """
    Return a new sorted array along the first axis using the mergesort algorithm.

    Parameters
    ----------
    a : Union[JaxArray, list, tuple]
        Input array or sequence to be sorted. It should have elements with
        comparable data types.
    out : JaxArray, optional
        An optional output array to store the sorted result. If provided,
        it must have the same shape as 'a'. Defaults to None.

    Returns
    -------
    JaxArray
        A new sorted array. The data type of the output is determined by the
        data type of the input 'a'.

    Examples
    --------

    With a list input:

    >>> lst = [3, 1, 2]
    >>> ivy.msort(lst)
    ivy.array([1, 2, 3])

    With a tuple input:

    >>> tpl = (3, 1, 2)
    >>> ivy.msort(tpl)
    ivy.array([1, 2, 3])

    Using an output array:

    >>> a = ivy.asarray([[8, 9, 6],[6, 2, 6]])
    >>> ivy.msort(a)
    ivy.array(
    [[6, 2, 6],
     [8, 9, 6]]
    )
    """
    return jnp.sort(a, axis=0, kind="mergesort")
