# global
from numbers import Number
from typing import Union, Optional, List, Sequence

import numpy as np

# local
import ivy
from ivy.functional.backends.numpy.device import _to_device
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
)
from .data_type import as_native_dtype


# Array API Standard #
# -------------------#


def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[np.dtype] = None,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype:
        dtype = as_native_dtype(dtype)
    res = _to_device(np.arange(start, stop, step, dtype=dtype), device=device)
    if not dtype:
        if res.dtype == np.float64:
            return res.astype(np.float32)
        elif res.dtype == np.int64:
            return res.astype(np.int32)
    return res


@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
def asarray(
    obj: Union[
        np.ndarray, bool, int, float, tuple, NestedSequence, SupportsBufferProtocol
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[np.dtype] = None,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        if dtype is not None:
            obj = ivy.astype(obj, dtype, copy=False).to_native()
        ret = np.copy(obj) if copy else obj
        return _to_device(ret, device=device)
    elif isinstance(obj, (list, tuple, dict)) and len(obj) != 0 and dtype is None:
        dtype = ivy.default_dtype(item=obj, as_native=True)
    else:
        dtype = ivy.default_dtype(dtype=dtype, item=obj, as_native=True)
    if copy is True:
        return _to_device(np.copy(np.asarray(obj, dtype=dtype)), device=device)
    else:
        return _to_device(np.asarray(obj, dtype=dtype), device=device)


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: np.dtype,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return _to_device(np.empty(shape, dtype), device=device)


def empty_like(
    x: np.ndarray, /, *, dtype: np.dtype, device: str, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return _to_device(np.empty_like(x, dtype=dtype), device=device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: np.dtype,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_cols is None:
        n_cols = n_rows
    i = np.eye(n_rows, n_cols, k, dtype)
    if batch_shape is None:
        return _to_device(i, device=device)
    else:
        reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
        tile_dims = list(batch_shape) + [1, 1]
        return_mat = np.tile(np.reshape(i, reshape_dims), tile_dims)
        return _to_device(return_mat, device=device)


def from_dlpack(x, /, *, out: Optional[np.ndarray] = None):
    return np.from_dlpack(x)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, np.dtype]] = None,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    dtype = ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True)
    ivy.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
    return _to_device(
        np.full(shape, fill_value, dtype),
        device=device,
    )


def full_like(
    x: np.ndarray,
    /,
    fill_value: Number,
    *,
    dtype: np.dtype,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ivy.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
    return _to_device(np.full_like(x, fill_value, dtype=dtype), device=device)


def linspace(
    start: Union[np.ndarray, float],
    stop: Union[np.ndarray, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: np.dtype,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if axis is None:
        axis = -1
    ans = np.linspace(start, stop, num, endpoint, dtype=dtype, axis=axis)
    # Waiting for fix when start is -0.0: https://github.com/numpy/numpy/issues/21513
    if (
        ans.shape[0] >= 1
        and (not isinstance(start, np.ndarray))
        and (not isinstance(stop, np.ndarray))
    ):
        ans[0] = start
    return _to_device(ans, device=device)


def meshgrid(
    *arrays: np.ndarray,
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    return np.meshgrid(*arrays, sparse=sparse, indexing=indexing)


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: np.dtype,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return _to_device(np.ones(shape, dtype), device=device)


def ones_like(
    x: np.ndarray, /, *, dtype: np.dtype, device: str, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return _to_device(np.ones_like(x, dtype=dtype), device=device)


def tril(
    x: np.ndarray, /, *, k: int = 0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.tril(x, k)


def triu(
    x: np.ndarray, /, *, k: int = 0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.triu(x, k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: np.dtype,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return _to_device(np.zeros(shape, dtype), device=device)


def zeros_like(
    x: np.ndarray, /, *, dtype: np.dtype, device: str, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return _to_device(np.zeros_like(x, dtype=dtype), device=device)


# Extra #
# ------#


array = asarray


def copy_array(
    x: np.ndarray,
    *,
    to_ivy_array: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if to_ivy_array:
        return ivy.to_ivy(x.copy())
    return x.copy()


def one_hot(
    indices: np.ndarray,
    depth: int,
    /,
    *,
    on_value: Optional[Number] = None,
    off_value: Optional[Number] = None,
    axis: Optional[int] = None,
    dtype: Optional[np.dtype] = None,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    on_none = on_value is None
    off_none = off_value is None

    if dtype is None:
        if on_none and off_none:
            dtype = np.float32
        else:
            if not on_none:
                dtype = np.array(on_value).dtype
            elif not off_none:
                dtype = np.array(off_value).dtype

    res = np.eye(depth, dtype=dtype)[np.array(indices, dtype="int64").reshape(-1)]
    res = res.reshape(list(indices.shape) + [depth])

    if not on_none and not off_none:
        res = np.where(res == 1, on_value, off_value)

    if axis is not None:
        res = np.moveaxis(res, -1, axis)

    return res


def frombuffer(
    buffer: bytes,
    dtype: Optional[np.dtype] = float,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> np.ndarray:
    if isinstance(dtype, list):
        dtype = np.dtype(dtype[0])
    return np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
