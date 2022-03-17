# global
from typing import Tuple
import numpy as np


def unique_inverse(x: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    values, inverse_indices = np.unique(x, return_inverse=True)
    t = Tuple[values, inverse_indices]
    return t
