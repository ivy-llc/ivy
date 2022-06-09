# global
import numpy as np
from typing import Tuple, Union, Optional

# local
import ivy

# Array API Standard #
# -------------------#


def min(
    x: np.ndarray, axis: Union[int, Tuple[int]] = None, keepdims: bool = False
) -> np.ndarray:
    return np.asarray(np.amin(a=x, axis=axis, keepdims=keepdims))


def max(
    x: np.ndarray, axis: Union[int, Tuple[int]] = None, keepdims: bool = False
) -> np.ndarray:
    return np.asarray(np.amax(a=x, axis=axis, keepdims=keepdims))


def var(
    x: np.ndarray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> np.ndarray:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return np.asarray(np.var(x, axis=axis, keepdims=keepdims))


def sum(
    x: np.ndarray,
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    *,
    dtype: np.dtype = None
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
    return np.asarray(np.sum(a=x, axis=axis, dtype=dtype, keepdims=keepdims))


def prod(
    x: np.ndarray,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: np.dtype = None,
    keepdims: bool = False,
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
    return np.asarray(np.prod(a=x, axis=axis, dtype=dtype, keepdims=keepdims))


def mean(
    x: np.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return np.asarray(np.mean(x, axis=axis, keepdims=keepdims))


def std(
    x: np.ndarray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> np.ndarray:
    return np.asarray(np.std(x, axis=axis, ddof=correction, keepdims=keepdims))


# Extra #
# ------#


def einsum(equation: str, *operands: np.ndarray) -> np.ndarray:
    return np.asarray(np.einsum(equation, *operands))
