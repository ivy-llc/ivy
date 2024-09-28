# global
from numbers import Number
from typing import Union, Optional, List, Sequence, Tuple

import numpy as np

# local
import ivy
from ivy.functional.ivy.creation import (
    _asarray_to_native_arrays_and_back,
    _asarray_infer_device,
    _asarray_infer_dtype,
    _asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
    _asarray_inputs_to_native_shapes,
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
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype:
        dtype = as_native_dtype(dtype)
    res = np.arange(start, stop, step, dtype=dtype)
    if not dtype:
        if res.dtype == np.float64:
            return res.astype(np.float32)
        elif res.dtype == np.int64:
            return res.astype(np.int32)
    return res


def complex(
    real: np.ndarray, imag: np.ndarray, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return real + imag * 1j


@_asarray_to_native_arrays_and_back
@_asarray_infer_device
@_asarray_handle_nestable
@_asarray_inputs_to_native_shapes
@_asarray_infer_dtype
def asarray(
    obj: Union[
        np.ndarray, bool, int, float, tuple, NestedSequence, SupportsBufferProtocol
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[np.dtype] = None,
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.asarray(obj, dtype=dtype)
    return np.copy(ret) if copy else ret


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: np.dtype,
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.empty(shape, dtype)


def empty_like(
    x: np.ndarray,
    /,
    *,
    dtype: np.dtype,
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.empty_like(x, dtype=dtype)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: np.dtype,
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_cols is None:
        n_cols = n_rows
    i = np.eye(n_rows, n_cols, k, dtype)
    if batch_shape is None:
        return i
    else:
        reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
        tile_dims = list(batch_shape) + [1, 1]
        return_mat = np.tile(np.reshape(i, reshape_dims), tile_dims)
        return return_mat


def to_dlpack(x, /, *, out: Optional[np.ndarray] = None):
    return x.__dlpack__()


class _dlpack_wrapper:
    def __init__(self, capsule) -> None:
        self.capsule = capsule

    def dlpack(self):
        return self.capsule


def from_dlpack(x, /, *, out: Optional[np.ndarray] = None):
    if not hasattr(x, "__dlpack__"):
        capsule = _dlpack_wrapper(x)
    else:
        capsule = x
    return np.from_dlpack(capsule)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, np.dtype]] = None,
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    dtype = ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True)
    return np.full(shape, fill_value, dtype)


def full_like(
    x: np.ndarray,
    /,
    fill_value: Number,
    *,
    dtype: np.dtype,
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.full_like(x, fill_value, dtype=dtype)


def linspace(
    start: Union[np.ndarray, float],
    stop: Union[np.ndarray, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: np.dtype,
    device: Optional[str] = None,
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
    return ans


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
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.ones(shape, dtype)


def ones_like(
    x: np.ndarray,
    /,
    *,
    dtype: np.dtype,
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.ones_like(x, dtype=dtype)


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
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.zeros(shape, dtype)


def zeros_like(
    x: np.ndarray,
    /,
    *,
    dtype: np.dtype,
    device: Optional[str] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.zeros_like(x, dtype=dtype)


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
    device: Optional[str] = None,
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


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: Optional[str] = None,
) -> Tuple[np.ndarray]:
    return tuple(np.asarray(np.triu_indices(n=n_rows, k=k, m=n_cols)))
