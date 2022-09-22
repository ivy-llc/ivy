from typing import Optional, Tuple

import ivy
import numpy as np


# Array API Standard #
# ------------------ #


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
    *,
    as_tuple: bool = True,
) -> Tuple[np.ndarray]:
    if as_tuple:
        return np.nonzero(x)
    else:
        return np.stack(np.nonzero(x), axis=1)


def where(
    condition: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.where(condition, x1, x2).astype(x1.dtype)


# Extra #
# ----- #


def argwhere(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.argwhere(x)
