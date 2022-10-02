from numbers import Number
from typing import Optional, Tuple, Union

import cupy as cp

import ivy


# Array API Standard #
# ------------------ #


def argmax(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    ret = cp.argmax(x, axis=axis, keepdims=keepdims, out=out)
    return cp.array(ret)


argmax.support_native_out = True


def argmin(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    ret = cp.argmin(x, axis=axis, keepdims=keepdims, out=out)
    return cp.array(ret)


argmin.support_native_out = True


def nonzero(
    x: cp.ndarray,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[cp.ndarray, Tuple[cp.ndarray]]:
    res = cp.nonzero(x)

    if size is not None:
        if isinstance(fill_value, float):
            res = cp.asarray(res, dtype=cp.float64)

        diff = size - res[0].shape[0]
        if diff > 0:
            res = cp.pad(res, ((0, 0), (0, diff)), constant_values=fill_value)
        elif diff < 0:
            res = cp.array(res)[:, :size]

    if as_tuple:
        return tuple(res)
    return cp.stack(res, axis=1)


def where(
    condition: cp.ndarray,
    x1: cp.ndarray,
    x2: cp.ndarray,
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.where(condition, x1, x2).astype(x1.dtype)


# Extra #
# ----- #


def argwhere(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.argwhere(x)
