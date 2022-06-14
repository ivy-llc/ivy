import numpy as np

from typing import Optional, Tuple


def argmax(
    x: np.ndarray,
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.argmax(x, axis=axis, keepdims=keepdims)


def argmin(
    x: np.ndarray,
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.argmin(x, axis=axis, keepdims=keepdims)


def nonzero(x: np.ndarray) -> Tuple[np.ndarray]:
    return np.nonzero(x)


def where(condition: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.where(condition, x1, x2)
