from typing import Optional
import numpy as np


def diagflat(
    x: np.ndarray,
    /,
    *,
    k: int = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.diagflat(x, k=k)


def kron(
    a: np.ndarray,
    b: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.kron(a, b)


kron.support_native_out = False
