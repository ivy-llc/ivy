# global
import cupy as cp
from typing import Union, Optional, Sequence

# local
import ivy


# Array API Standard #
# -------------------#


def min(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return cp.asarray(cp.amin(a=x, axis=axis, keepdims=keepdims, out=out))


min.support_native_out = True


def max(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return cp.asarray(cp.amax(a=x, axis=axis, keepdims=keepdims, out=out))


max.support_native_out = True


def mean(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return cp.asarray(cp.mean(x, axis=axis, keepdims=keepdims, out=out)).astype(x.dtype)


mean.support_native_out = True


def _infer_dtype(dtype: cp.dtype):
    default_dtype = ivy.infer_default_dtype(dtype)
    if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
        return default_dtype
    return dtype


def prod(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[cp.dtype] = None,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return cp.asarray(cp.prod(a=x, axis=axis, dtype=dtype, keepdims=keepdims, out=out))


prod.support_native_out = True


def std(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return cp.asarray(cp.std(x, axis=axis, ddof=correction, keepdims=keepdims, out=out))


std.support_native_out = True


def sum(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[cp.dtype] = None,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return cp.asarray(
        cp.sum(
            a=x,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            out=out,
        )
    )


sum.support_native_out = True


def var(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if axis is None:
        axis = tuple(range(len(x.shape)))
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if isinstance(correction, int):
        return cp.asarray(
            cp.var(x, axis=axis, ddof=correction, keepdims=keepdims, out=out)
        ).astype(x.dtype)
    if x.size == 0:
        return cp.asarray(float("nan"))
    size = 1
    for a in axis:
        size *= x.shape[a]
    return cp.asarray(
        cp.multiply(
            cp.var(x, axis=axis, keepdims=keepdims, out=out),
            ivy.stable_divide(size, (size - correction)),
        )
    ).astype(x.dtype)


var.support_native_out = True


# Extra #
# ------#


def cumprod(
    x: cp.ndarray,
    axis: int = 0,
    exclusive: bool = False,
    *,
    dtype: Optional[cp.dtype] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    if exclusive:
        x = cp.swapaxes(x, axis, -1)
        x = cp.concatenate((cp.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = cp.cumprod(x, -1, dtype=dtype)
        res = cp.swapaxes(res, axis, -1)
        if out is not None:
            return ivy.inplace_update(out, res)
        return res
    return cp.cumprod(x, axis, dtype=dtype, out=out)


cumprod.support_native_out = True


def cumsum(
    x: cp.ndarray,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    dtype: Optional[cp.dtype] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if dtype is None:
        if x.dtype == "bool":
            dtype = ivy.default_int_dtype(as_native=True)
        else:
            dtype = _infer_dtype(x.dtype)
    if exclusive or reverse:
        if exclusive and reverse:
            x = cp.cumsum(cp.flip(x, axis=axis), axis=axis, dtype=dtype)
            x = cp.swapaxes(x, axis, -1)
            x = cp.concatenate((cp.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = cp.swapaxes(x, axis, -1)
            res = cp.flip(x, axis=axis)
        elif exclusive:
            x = cp.swapaxes(x, axis, -1)
            x = cp.concatenate((cp.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = cp.cumsum(x, -1, dtype=dtype)
            res = cp.swapaxes(x, axis, -1)
        elif reverse:
            x = cp.cumsum(cp.flip(x, axis=axis), axis=axis, dtype=dtype)
            res = cp.flip(x, axis=axis)
        return res
    return cp.cumsum(x, axis, dtype=dtype, out=out)


cumsum.support_native_out = True


def einsum(
    equation: str, *operands: cp.ndarray, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.asarray(cp.einsum(equation, *operands, out=out))


einsum.support_native_out = True
