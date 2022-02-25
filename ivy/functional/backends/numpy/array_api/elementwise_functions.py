# global
import numpy as np
import numpy.array_api as npa


def isfinite(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.isfinite(npa.asarray(x)))


def cos(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.cos(npa.asarray(x)))


def logical_not(x: np.ndarray) -> np.ndarray:
    return np.logical_not(x)
