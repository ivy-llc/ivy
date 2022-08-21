from typing import Optional, Tuple

import numpy as np


def argmax(
    x: np.ndarray,
    axis: Optional[int] = None,
    keepdims: bool = False,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.argmax(x, axis=axis, keepdims=keepdims, out=out)
    ret = np.array(ret)
    return ret


argmax.support_native_out = True


def argmin(
    x: np.ndarray,
    axis: Optional[int] = None,
    keepdims: bool = False,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.argmin(x, axis=axis, keepdims=keepdims, out=out)
    ret = np.array(ret)
    return ret


argmin.support_native_out = True


def nonzero(x: np.ndarray) -> Tuple[np.ndarray]:
    return np.nonzero(x)


def where(
    condition: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    dtype = np.promote_types(x1.dtype, x2.dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.where(condition, x1, x2)
