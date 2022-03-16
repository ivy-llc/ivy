# global
import numpy as np
from typing import Tuple, Union


# Array API Standard #
# -------------------#

def min(x: np.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> np.ndarray:
    return np.amin(a=x, axis=axis, keepdims=keepdims)


def max(x: np.ndarray,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> np.ndarray:
    return np.amax(a=x, axis=axis, keepdims=keepdims)


# Extra #
# ------#
