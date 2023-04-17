import numpy as np
from typing import Optional

def difference(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.setdiff1d(x1, x2)


difference.support_native_out = False
