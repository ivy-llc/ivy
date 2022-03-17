# global
from typing import Tuple
import numpy as np


def unique_inverse(x: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    values, inverse_indices = np.unique(x, return_inverse=True)
    return values, inverse_indices
