# global
import cupy as cp
from typing import Optional


def argsort(
    x: cp.ndarray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x = -1 * cp.searchsorted(cp.unique(x), x) if descending else x
    kind = "stable" if stable else "quicksort"
    return cp.argsort(x, axis, kind=kind)


def sort(
    x: cp.ndarray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    kind = "stable" if stable else "quicksort"
    ret = cp.asarray(cp.sort(x, axis=axis, kind=kind))
    if descending:
        ret = cp.asarray((cp.flip(ret, axis)))
    return ret


def searchsorted(
    x: cp.ndarray,
    v: cp.ndarray,
    /,
    *,
    side="left",
    sorter=None,
    ret_dtype=cp.int64,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.searchsorted(x, v, side=side, sorter=sorter).astype(ret_dtype)
