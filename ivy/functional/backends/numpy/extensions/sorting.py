# global
from typing import Optional, Union

import numpy as np


# msort
def msort(
    a: Union[np.ndarray, list, tuple], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.msort(a)


msort_support_native_out = False
