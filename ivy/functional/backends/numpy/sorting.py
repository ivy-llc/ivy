# global
import numpy as np
from typing import Optional, Literal, Union, List

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def argsort(
    x: np.ndarray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    kind = "stable" if stable else "quicksort"
    return (
        np.argsort(-x, axis=axis, kind=kind)
        if descending
        else np.argsort(x, axis=axis, kind=kind)
    )


def sort(
    x: np.ndarray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    kind = "stable" if stable else "quicksort"
    ret = np.asarray(np.sort(x, axis=axis, kind=kind))
    if descending:
        ret = np.asarray((np.flip(ret, axis)))
    return ret


# msort
@with_unsupported_dtypes({"1.25.2 and below": ("complex",)}, backend_version)
def msort(
    a: Union[np.ndarray, list, tuple], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.msort(a)


msort.support_native_out = False


def searchsorted(
    x: np.ndarray,
    v: np.ndarray,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Optional[Union[np.ndarray, List[int]]] = None,
    ret_dtype: np.dtype = np.int64,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    assert ivy.is_int_dtype(ret_dtype), ValueError(
        "only Integer data types are supported for ret_dtype."
    )
    is_sorter_provided = sorter is not None
    if is_sorter_provided:
        assert ivy.is_int_dtype(sorter.dtype) and not ivy.is_uint_dtype(
            sorter.dtype
        ), TypeError(
            f"Only signed integer data type for sorter is allowed, got {sorter.dtype}."
        )
    if x.ndim != 1:
        assert x.shape[:-1] == v.shape[:-1], RuntimeError(
            "the first N-1 dimensions of x array and v array "
            f"must match, got {x.shape} and {v.shape}"
        )
        if is_sorter_provided:
            x = np.take_along_axis(x, sorter, axis=-1)
        original_shape = v.shape
        x = x.reshape(-1, x.shape[-1])
        v = v.reshape(-1, v.shape[-1])
        out_array = np.empty_like(v)
        for i in range(x.shape[0]):
            out_array[i] = np.searchsorted(x[i], v[i], side=side)
        ret = out_array.reshape(original_shape)
    else:
        ret = np.searchsorted(x, v, side=side, sorter=sorter)
    return ret.astype(ret_dtype)
