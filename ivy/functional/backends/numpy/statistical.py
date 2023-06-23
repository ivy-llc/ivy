# global
import math
import numpy as np
from typing import Union, Optional, Sequence, Tuple

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


@with_unsupported_dtypes({"1.25.0 and below": "bfloat16"}, backend_version)
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


@with_unsupported_dtypes({"1.25.0 and below": "bfloat16"}, backend_version)
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


def cummax(
    x: np.ndarray,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if x.dtype in (np.bool_, np.float16):
        x = x.astype(np.float64)
    elif x.dtype in (np.int16, np.int8, np.uint8):
        x = x.astype(np.int64)
    elif x.dtype in (np.complex128, np.complex64):
        x = np.real(x).astype(np.float64)

    if exclusive or reverse:
        if exclusive and reverse:
            indices = __find_cummax_indices(np.flip(x, axis=axis), axis=axis)
            x = np.maximum.accumulate(np.flip(x, axis=axis), axis=axis, dtype=x.dtype)
            x = np.swapaxes(x, axis, -1)
            indices = np.swapaxes(indices, axis, -1)
            x, indices = np.concatenate(
                (np.zeros_like(x[..., -1:]), x[..., :-1]), -1
            ), np.concatenate((np.zeros_like(indices[..., -1:]), indices[..., :-1]), -1)
            x, indices = np.swapaxes(x, axis, -1), np.swapaxes(indices, axis, -1)
            res, indices = np.flip(x, axis=axis), np.flip(indices, axis=axis)

        elif exclusive:
            x = np.swapaxes(x, axis, -1)
            x = np.concatenate((np.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = np.swapaxes(x, axis, -1)
            indices = __find_cummax_indices(x, axis=axis)
            res = np.maximum.accumulate(x, axis=axis, dtype=x.dtype)
        elif reverse:
            x = np.flip(x, axis=axis)
            indices = __find_cummax_indices(x, axis=axis)
            x = np.maximum.accumulate(x, axis=axis)
            res, indices = np.flip(x, axis=axis), np.flip(indices, axis=axis)
        return res, indices
    indices = __find_cummax_indices(x, axis=axis)
    return np.maximum.accumulate(x, axis=axis, dtype=x.dtype), indices


cummax.support_native_out = True


def __find_cummax_indices(
    x: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    indices = []
    if type(x[0]) == np.ndarray:
        if axis >= 1:
            for ret1 in x:
                indice = __find_cummax_indices(ret1, axis=axis - 1)
                indices.append(indice)

        else:
            indice_list = __get_index(x.tolist())
            indices, n1 = x.copy(), {}
            indices.fill(0)
            indice_list = sorted(indice_list, key=lambda i: i[1])
            for y, y_index in indice_list:
                multi_index = y_index
                if tuple(multi_index[1:]) not in n1:
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices[y_index] = multi_index[0]
                elif (
                    y >= x[tuple([n1[tuple(multi_index[1:])]] + list(multi_index[1:]))]
                ):
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices[y_index] = multi_index[0]
                else:
                    indices[y_index] = n1[tuple(multi_index[1:])]
    else:
        n = 0
        for index1, ret1 in enumerate(x):
            if x[n] <= ret1 or index1 == 0:
                n = index1
            indices.append(n)
    return np.array(indices, dtype=np.int64)


def __get_index(lst, indices=None, prefix=None):
    if indices is None:
        indices = []
    if prefix is None:
        prefix = []

    if isinstance(lst, list):
        for i, sub_lst in enumerate(lst):
            sub_indices = prefix + [i]
            __get_index(sub_lst, indices, sub_indices)
    else:
        indices.append((lst, tuple(prefix)))
    return indices


@_scalar_output_to_0d_array
def einsum(
    equation: str, *operands: np.ndarray, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.einsum(equation, *operands, out=out)


einsum.support_native_out = True


def igamma(
    a: np.ndarray,
    /,
    *,
    x: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    def igamma_cal(a, x):
        t = np.linspace(0, x, 10000, dtype=np.float64)
        y = np.exp(-t) * (t ** (a - 1))
        integral = np.trapz(y, t)
        return np.float32(integral / math.gamma(a))

    igamma_vec = np.vectorize(igamma_cal)
    return igamma_vec(a, x)
