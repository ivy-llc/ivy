import numpy as np
from typing import Tuple


def intersection(
        ar1: np.ndarray,
        ar2: np.ndarray,
        /,
        *,
        assume_unique: bool = False,
        return_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
