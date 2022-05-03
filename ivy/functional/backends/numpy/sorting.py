# global
import numpy as np
from typing import Optional

# local
import ivy


def argsort(
    x: np.ndarray,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if descending:
        ret = np.asarray(
            np.argsort(-1 * np.searchsorted(np.unique(x), x), axis, kind="stable")
        )
    else:
        ret = np.asarray(np.argsort(x, axis, kind="stable"))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sort(
    x: np.ndarray,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    kind = "stable" if stable else "quicksort"
    ret = np.asarray(np.sort(x, axis=axis, kind=kind))
    if descending:
        ret = np.asarray((np.flip(ret, axis)))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret
