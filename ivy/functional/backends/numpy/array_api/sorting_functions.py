# global
import numpy as np


def argsort(x: np.ndarray,
            axis: int = -1,
            descending: bool = False,
            stable: bool = True)\
            -> np.ndarray:
    if descending:
        return np.asarray(np.argsort(-1 * np.searchsorted(np.unique(x), x), axis, kind='stable'))
    else:
        return np.asarray(np.argsort(x, axis, kind='stable'))
