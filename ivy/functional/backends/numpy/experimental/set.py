import numpy as np
from typing import Optional


def union(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.union1d(x1, x2)


union.support_native_out = True