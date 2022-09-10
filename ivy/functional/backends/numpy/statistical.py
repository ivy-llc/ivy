# global
import numpy as np
from typing import Tuple, Union, Optional, Sequence

# local
import ivy


# Array API Standard #
# -------------------#


def max(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.amax(a=x, axis=axis, keepdims=keepdims, out=out))


max.support_native_out = True


def mean(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.mean(x, axis=axis, keepdims=keepdims, out=out)).astype(x.dtype)


mean.support_native_out = True


def min(
    x: np.ndarray,
    /,
    *,
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.amin(a=x, axis=axis, keepdims=keepdims, out=out))


min.support_native_out = True


def _infer_dtype(dtype: np.dtype):
    default_dtype = ivy.infer_default_dtype(dtype)
    if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
        return default_dtype
    return dtype


def prod(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[np.dtype] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.prod(a=x, axis=axis, dtype=dtype, keepdims=keepdims, out=out))


prod.support_native_out = True


def std(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.std(x, axis=axis, ddof=correction, keepdims=keepdims, out=out))


std.support_native_out = True


def sum(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[np.dtype] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(
        np.sum(
            a=x,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            out=out,
        )
    )


sum.support_native_out = True


def var(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if axis is None:
        axis = tuple(range(len(x.shape)))
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if isinstance(correction, int):
        return np.asarray(
            np.var(x, axis=axis, ddof=correction, keepdims=keepdims, out=out)
        ).astype(x.dtype)
    size = 1
    for a in axis:
        size *= x.shape[a]
    return np.asarray(
        np.var(x, axis=axis, keepdims=keepdims, out=out) * (size / (size - correction))
    ).astype(x.dtype)


var.support_native_out = True


# Extra #
# ------#


def cumprod(
    x: np.ndarray,
    axis: int = 0,
    exclusive: bool = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    if exclusive:
        x = np.swapaxes(x, axis, -1)
        x = np.concatenate((np.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = np.cumprod(x, -1, dtype=dtype)
        res = np.swapaxes(res, axis, -1)
        if out is not None:
            return ivy.inplace_update(out, res)
        return res
    return np.cumprod(x, axis, dtype=dtype, out=out)


cumprod.support_native_out = True


def cumsum(
    x: np.ndarray,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None:
        if x.dtype == "bool":
            dtype = ivy.default_int_dtype(as_native=True)
        else:
            dtype = _infer_dtype(x.dtype)
    if exclusive or reverse:
        if exclusive and reverse:
            x = np.cumsum(np.flip(x, axis=axis), axis=axis, dtype=dtype)
            x = np.swapaxes(x, axis, -1)
            x = np.concatenate((np.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = np.swapaxes(x, axis, -1)
            res = np.flip(x, axis=axis)
        elif exclusive:
            x = np.swapaxes(x, axis, -1)
            x = np.concatenate((np.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = np.cumsum(x, -1, dtype=dtype)
            res = np.swapaxes(x, axis, -1)
        elif reverse:
            x = np.cumsum(np.flip(x, axis=axis), axis=axis, dtype=dtype)
            res = np.flip(x, axis=axis)
        return res
    return np.cumsum(x, axis, dtype=dtype, out=out)


cumsum.support_native_out = True


def einsum(
    equation: str, *operands: np.ndarray, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.asarray(np.einsum(equation, *operands, out=out))


einsum.support_native_out = True
