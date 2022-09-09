from typing import Optional, Tuple

import ivy
import numpy as np


def argmax(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.argmax(x, axis=axis, keepdims=keepdims, out=out)
    return np.array(ret)


argmax.support_native_out = True


def argmin(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.argmin(x, axis=axis, keepdims=keepdims, out=out)
    return np.array(ret)


argmin.support_native_out = True


def nonzero(
    x: np.ndarray,
    /,
) -> Tuple[np.ndarray]:
    return np.nonzero(x)


def where(
    condition: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.where(condition, x1, x2)
