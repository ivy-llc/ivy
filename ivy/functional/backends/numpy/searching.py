from typing import Optional, Tuple, Union

import numpy as np

import ivy


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
    size: Optional[int] = None,
    fill_value: int = 0,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    res = np.nonzero(x)

    if size is not None:
        diff = size - res[0].shape[0]
        if diff > 0:
            res = np.pad(res, ((0, 0), (0, diff)), constant_values=fill_value)
        elif diff < 0:
            res = np.array(res)[:, :size]

    if as_tuple:
        return tuple(res)
    return np.stack(res, axis=1)


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
