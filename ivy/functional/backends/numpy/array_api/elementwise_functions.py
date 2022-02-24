# global
import numpy as np


def isfinite(x: np.ndarray)\
        -> np.ndarray:
    return np.isfinite(x)


def equal(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return x1 == x2


def less_equal(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return x1 <= x2
