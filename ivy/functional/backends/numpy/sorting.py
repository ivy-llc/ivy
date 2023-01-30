# global
import numpy as np
from typing import Optional

# local
import ivy


def argsort(
    x: np.ndarray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x = -1 * np.searchsorted(np.unique(x), x) if descending else x
    kind = "stable" if stable else "quicksort"
    return np.argsort(x, axis, kind=kind)


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


def searchsorted(
    x: np.ndarray,
    v: np.ndarray,
    /,
    *,
    side="left",
    sorter=None,
    ret_dtype=np.int64,
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
            f"the first N-1 dimensions of x array and v array "
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


def lexsort(
    x: np.ndarray,
    /,
    *,
    axis: int = -1,
) -> np.ndarray:
    return np.lexsort(x, axis)
