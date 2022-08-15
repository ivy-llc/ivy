#for review

# global
import numpy as np
from typing import Tuple, Union, Optional, Sequence

# local
import ivy


# Array API Standard #
# -------------------#


def max(
    x: np.ndarray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.amax(a=x, axis=axis, keepdims=keepdims, out=out))


max.support_native_out = True


def mean(
    x: np.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.mean(x, axis=axis, keepdims=keepdims, out=out))


mean.support_native_out = True


def min(
    x: np.ndarray,
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.amin(a=x, axis=axis, keepdims=keepdims, out=out))


min.support_native_out = True


def prod(
    x: np.ndarray,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[np.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None and np.issubdtype(x.dtype, np.integer):
        if np.issubdtype(x.dtype, np.signedinteger) and x.dtype in [
            np.int8,
            np.int16,
            np.int32,
        ]:
            dtype = np.int32
        elif np.issubdtype(x.dtype, np.unsignedinteger) and x.dtype in [
            np.uint8,
            np.uint16,
            np.uint32,
        ]:
            dtype = np.uint32
        elif x.dtype == np.int64:
            dtype = np.int64
        else:
            dtype = np.uint64
    dtype = ivy.as_native_dtype(dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.prod(a=x, axis=axis, dtype=dtype, keepdims=keepdims, out=out))


prod.support_native_out = True


def std(
    x: np.ndarray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.std(x, axis=axis, ddof=correction, keepdims=keepdims, out=out))


std.support_native_out = True


def sum(
    x: np.ndarray,
    *,
    axis: Union[int, Tuple[int]] = None,
    dtype: np.dtype = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype is None and np.issubdtype(x.dtype, np.integer):
        if np.issubdtype(x.dtype, np.signedinteger) and x.dtype in [
            np.int8,
            np.int16,
            np.int32,
        ]:
            dtype = np.int32
        elif np.issubdtype(x.dtype, np.unsignedinteger) and x.dtype in [
            np.uint8,
            np.uint16,
            np.uint32,
        ]:
            dtype = np.uint32
        elif x.dtype == np.int64:
            dtype = np.int64
        else:
            dtype = np.uint64
    dtype = ivy.as_native_dtype(dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.sum(a=x, axis=axis, dtype=dtype, keepdims=keepdims, out=out))


sum.support_native_out = True


def var(
    x: np.ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: Optional[bool] = False,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.asarray(np.var(x, axis=axis, ddof=correction, keepdims=keepdims, out=out))


var.support_native_out = True


# Extra #
# ------#


def einsum(
    equation: str, *operands: np.ndarray, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.asarray(np.einsum(equation, *operands, out=out))


einsum.support_native_out = True
