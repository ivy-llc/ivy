# global
import numpy as np
from typing import Optional, Union


# msort
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
