import numpy as np
from typing import Tuple


def intersection(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    assume_unique: bool = False,
    return_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.intersect1d(
        x1, x2, assume_unique=assume_unique, return_indices=return_indices
    )
