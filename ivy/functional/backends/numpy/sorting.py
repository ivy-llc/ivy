# global
import numpy as np
from typing import Optional


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
    v = list(v)
    return np.searchsorted(x, v, side=side, sorter=sorter).astype(ret_dtype)
