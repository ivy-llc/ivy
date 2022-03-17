# global
import numpy as np
from typing import Tuple
from collections import namedtuple


def unique_inverse(x: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    values, inverse_indices = np.unique(x, return_inverse=True)
    return out(values, inverse_indices)
