# global
import numpy as np
from typing import Union, Optional, Sequence

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from . import backend_version


# Array API Standard #
# -------------------#


def min(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.amin(a=x, axis=axis, keepdims=keepdims, out=out))


min.support_native_out = True


def max(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.amax(a=x, axis=axis, keepdims=keepdims, out=out))


max.support_native_out = True


@_scalar_output_to_0d_array
def mean(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return ivy.astype(
        np.mean(x, axis=axis, keepdims=keepdims, out=out), x.dtype, copy=False
    )


mean.support_native_out = True


def _infer_dtype(dtype: np.dtype):
    default_dtype = ivy.infer_default_dtype(dtype)
    if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
        return default_dtype
    return dtype


def prod(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[np.dtype] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.prod(a=x, axis=axis, dtype=dtype, keepdims=keepdims, out=out))


prod.support_native_out = True


def std(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
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
    keepdims: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None and not ivy.is_bool_dtype(x):
        dtype = x.dtype
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


@_scalar_output_to_0d_array
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
        return ivy.astype(
            np.var(x, axis=axis, ddof=correction, keepdims=keepdims, out=out),
            x.dtype,
            copy=False,
        )
    if x.size == 0:
        return np.asarray(float("nan"))
    size = 1
    for a in axis:
        size *= x.shape[a]
    return ivy.astype(
        np.multiply(
            np.var(x, axis=axis, keepdims=keepdims, out=out),
            ivy.stable_divide(size, (size - correction)),
        ),
        x.dtype,
        copy=False,
    )


var.support_native_out = True


# Extra #
# ------#


@with_unsupported_dtypes({"1.23.0 and below": ("float16", "bfloat16")}, backend_version)
def cumprod(
    x: np.ndarray,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None:
        if x.dtype == "bool":
            dtype = ivy.default_int_dtype(as_native=True)
        else:
            dtype = _infer_dtype(x.dtype)
    if not (exclusive or reverse):
        return np.cumprod(x, axis, dtype=dtype, out=out)
    elif exclusive and reverse:
        x = np.cumprod(np.flip(x, axis=axis), axis=axis, dtype=dtype)
        x = np.swapaxes(x, axis, -1)
        x = np.concatenate((np.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = np.swapaxes(x, axis, -1)
        return np.flip(x, axis=axis)
    elif exclusive:
        x = np.swapaxes(x, axis, -1)
        x = np.concatenate((np.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = np.cumprod(x, -1, dtype=dtype)
        return np.swapaxes(x, axis, -1)
    elif reverse:
        x = np.cumprod(np.flip(x, axis=axis), axis=axis, dtype=dtype)
        return np.flip(x, axis=axis)


cumprod.support_native_out = True


@with_unsupported_dtypes({"1.23.0 and below": ("float16", "bfloat16")}, backend_version)
def cummax(
    x: np.ndarray,
    /,
    *,
    axis: int = 0,
    reverse: bool = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None:
        if x.dtype == "bool":
            dtype = ivy.default_int_dtype(as_native=True)
        else:
            dtype = _infer_dtype(x.dtype)
    if not (reverse):
        return np.maximum.accumulate(x, axis, dtype=dtype, out=out)
    elif reverse:
        x = np.maximum.accumulate(np.flip(x, axis=axis), axis=axis, dtype=dtype)
        return np.flip(x, axis=axis)


cummax.support_native_out = True


@with_unsupported_dtypes({"1.23.0 and below": ("float16", "bfloat16")}, backend_version)
def cummin(
    x: np.ndarray,
    /,
    *,
    axis: int = 0,
    reverse: bool = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None:
        if x.dtype == "bool":
            dtype = ivy.default_int_dtype(as_native=True)
        else:
            dtype = _infer_dtype(x.dtype)
    if not (reverse):
        return np.minimum.accumulate(x, axis, dtype=dtype, out=out)
    elif reverse:
        x = np.minimum.accumulate(np.flip(x, axis=axis), axis=axis, dtype=dtype)
        return np.flip(x, axis=axis)


cummin.support_native_out = True


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
        if ivy.is_int_dtype(x.dtype):
            dtype = ivy.promote_types(x.dtype, ivy.default_int_dtype(as_native=True))
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


@_scalar_output_to_0d_array
def einsum(
    equation: str, *operands: np.ndarray, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.einsum(equation, *operands, out=out)


einsum.support_native_out = True
