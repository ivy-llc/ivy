# global
import numpy as np
from typing import Optional, Union
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# invert_permutation
def invert_permutation(
    x: Union[np.ndarray, list, tuple],
    /,
) -> np.ndarray:
    sorted_indices = np.argsort(x)
    inverse = np.zeros_like(sorted_indices)
    inverse[sorted_indices] = np.arange(len(x))
    inverse_permutation = np.argsort(inverse)
    return inverse_permutation


# msort
@with_unsupported_dtypes({"1.23.0 and below": ("complex",)}, backend_version)
def msort(
    a: Union[np.ndarray, list, tuple], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.msort(a)


msort.support_native_out = False


# lexsort
def lexsort(
    keys: np.ndarray, /, *, axis: int = -1, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.asarray(np.lexsort(keys, axis=axis))


lexsort.support_native_out = False
