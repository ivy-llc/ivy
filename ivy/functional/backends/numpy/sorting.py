# global
import numpy as np
from typing import Optional


def argsort(
    x: np.ndarray,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    if descending:
        ret = np.asarray(
            np.argsort(-1 * np.searchsorted(np.unique(x), x), axis, kind="stable")
        )
    else:
        ret = np.asarray(np.argsort(x, axis, kind="stable"))
    return ret


def sort(
    x: np.ndarray,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    kind = "stable" if stable else "quicksort"
    ret = np.asarray(np.sort(x, axis=axis, kind=kind))
    if descending:
        ret = np.asarray((np.flip(ret, axis)))
    return ret


def searchsorted(
    x: np.ndarray,
    v: np.ndarray,
    side="left",
    sorter=None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    v = list(v)
    res = np.asarray(np.searchsorted(x, v, side=side))
    return res
