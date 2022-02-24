# global
import numpy as np
import numpy.array_api as npa
from typing import Union, Tuple, Optional, List


def argsort(x: np.ndarray,
            axis: int = -1,
            descending: bool = False,
            stable: bool = True)\
            -> np.ndarray:
    if descending:
        return np.asarray(np.argsort(-1 * np.searchsorted(np.unique(x), x), axis, kind='stable'))
    else:
        return np.asarray(np.argsort(x, axis, kind='stable'))
