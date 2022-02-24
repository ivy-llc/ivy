# global
import numpy as np
import numpy.array_api as npa
from typing import Union, Tuple, Optional, List


# noinspection PyShadowingBuiltins
def cross(x1: np.ndarray, x2: np.ndarray, /, *, axis: int = -1) -> np.ndarray:
    return np.asarray(npa.linalg.cross(np.asarray(x1), np.asarray(x2), axis=axis))
