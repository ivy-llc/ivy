# global
import numpy as np
from typing import Optional, Union


# msort
def msort(
    a: Union[np.ndarray, list, tuple], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.msort(a)


msort_support_native_out = False


# lexsort
def lexsort(
    keys: Union[np.ndarray, list, tuple], /, *, out: Optional[np.ndarray] = None,
    axis: int = -1
) -> np.ndarray:
    return np.lexsort(keys, axis)


lexsort_support_native_out = False
