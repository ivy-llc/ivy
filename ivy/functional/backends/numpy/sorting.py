# global
import numpy as np


def argsort(x: np.ndarray,
            axis: int = -1,
            descending: bool = False,
            stable: bool = True) \
        -> np.ndarray:
    if descending:
        return np.asarray(np.argsort(-1 * np.searchsorted(np.unique(x), x), axis, kind='stable'))
    else:
        return np.asarray(np.argsort(x, axis, kind='stable'))


def sort(x: np.ndarray,
         axis: int = -1,
         descending: bool = False,
         stable: bool = True) -> np.ndarray:
    kind = "stable" if stable else "quicksort"
    res = np.asarray(np.sort(x, axis=axis, kind=kind))
    if descending:
        return np.asarray((np.flip(res, axis)))
    return res


