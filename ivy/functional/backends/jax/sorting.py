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
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    x : JaxArray
        Input array to be sorted.
    axis : int, optional
        Axis along which to sort. If set to -1, the function must sort along
        the last axis. Default: -1.
    descending : bool, optional
        If True, sort the elements in descending order; if False (default),
        sort in ascending order.
    stable : bool, optional
        If True (default), use a stable sorting algorithm to maintain the
        relative order of x values which compare as equal. If False, the
        returned indices may or may not maintain the relative order of x
        values which compare as equal (i.e., the relative order of x values
        which compare as equal is implementation-dependent). Default: True.
    out : Optional[JaxArray], optional
        An optional output array, for writing the result to. It must have the
        same shape as x. Defaults to None.

    Returns
    -------
    JaxArray
        An array with the same dtype and shape as x, with the elements sorted
        along the given axis.

    Examples
    --------
    With ivy.Array input:

    >>> x = ivy.array([7, 8, 6])
    >>> y = ivy.sort(x)
    >>> print(y)
    ivy.array([6, 7, 8])

    Sorting along a specific axis:

    >>> x = ivy.array([[[8.9,0], [19,5]],[[6,0.3], [19,0.5]]])
    >>> y = ivy.sort(x, axis=1, descending=True, stable=False)
    >>> print(y)
    ivy.array([[[19. ,  5. ],[ 8.9,  0. ]],[[19. ,  0.5],[ 6. ,  0.3]]])

    Sorting in descending order:

    >>> x = ivy.array([1.5, 3.2, 0.7, 2.5])
    >>> y = ivy.zeros(5)
    >>> ivy.sort(x, descending=True, stable=False, out=y)
    >>> print(y)
    ivy.array([3.2, 2.5, 1.5, 0.7])

    Using an output array:

    >>> x = ivy.array([[1.1, 2.2, 3.3],[-4.4, -5.5, -6.6]])
    >>> ivy.sort(x, out=x)
    >>> print(x)
    ivy.array([[ 1.1,  2.2,  3.3],
                [-6.6, -5.5, -4.4]
             ])

    With ivy.Container input:

    >>> x = ivy.Container(a=ivy.array([8, 6, 6]),b=ivy.array([[9, 0.7], [0.4, 0]]))
    >>> y = ivy.sort(x, descending=True)
    >>> print(y)
    {
        a: ivy.array([8, 6, 6]),
        b: ivy.array([[9., 0.7], [0.4, 0.]])
    }


    >>> x = ivy.Container(a=ivy.array([3, 0.7, 1]),b=ivy.array([[4, 0.9], [0.6, 0.2]]))
    >>> y = ivy.sort(x, descending=False, stable=False)
    >>> print(y)
    {
        a: ivy.array([0.7, 1., 3.]),
        b: ivy.array([[0.9, 4.], [0.2, 0.6]])
    }

    """
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
    a: Union[JaxArray, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.sort(a, axis=0, kind="mergesort")
