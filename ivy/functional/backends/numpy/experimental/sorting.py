# global
import numpy as np
from typing import Optional, Union


# invert_permutation
def invert_permutation(
    x: Union[np.ndarray, list, tuple],
    /,
) -> np.ndarray:
    sorted_indices = np.argsort(x)
    inverse = np.zeros_like(sorted_indices)
    inverse[sorted_indices] = np.arange(len(x))
    return np.argsort(inverse)


# lexsort
def lexsort(
    keys: np.ndarray, /, *, axis: int = -1, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.asarray(np.lexsort(keys, axis=axis))


lexsort.support_native_out = False
