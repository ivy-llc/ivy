from numbers import Number
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
    dtype: Optional[np.dtype] = np.int64,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.argmin(x, axis=axis, keepdims=keepdims, out=out)
    # The returned array must have the default array index data type.
    if dtype is not None:
        if dtype not in (np.int32, np.int64):
            return np.array(ret, dtype=np.int32)
        else:
            return np.array(ret, dtype=dtype)
    else:
        if ret.dtype not in (np.int32, np.int64):
            return np.array(ret, dtype=np.int32)
        else:
            return np.array(ret, dtype=ret.dtype)


argmin.support_native_out = True


def nonzero(
    x: np.ndarray,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    res = np.nonzero(x)

    if size is not None:
        if isinstance(fill_value, float):
            res = np.asarray(res, dtype=np.float64)

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
