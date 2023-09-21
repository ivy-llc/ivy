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
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    select_last_index: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if select_last_index:
        x = np.flip(x, axis=axis)
        ret = np.argmax(x, axis=axis, keepdims=keepdims)
        if axis is not None:
            ret = np.array(x.shape[axis] - ret - 1)
        else:
            ret = np.array(x.size - ret - 1)
    else:
        ret = np.array(np.argmax(x, axis=axis, keepdims=keepdims))
    if dtype:
        dtype = ivy.as_native_dtype(dtype)
        ret = ret.astype(dtype)
    return ret


def argmin(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
    select_last_index: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if select_last_index:
        x = np.flip(x, axis=axis)
        ret = np.argmin(x, axis=axis, keepdims=keepdims)
        if axis is not None:
            ret = np.array(x.shape[axis] - ret - 1)
        else:
            ret = np.array(x.size - ret - 1)
    else:
        ret = np.array(np.argmin(x, axis=axis, keepdims=keepdims))
    if dtype:
        dtype = ivy.as_native_dtype(dtype)
        return ret.astype(dtype)
    return ret


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
    return ivy.astype(np.where(condition, x1, x2), x1.dtype, copy=False)


# Extra #
# ----- #


def argwhere(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.argwhere(x)
