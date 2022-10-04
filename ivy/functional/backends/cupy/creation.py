# global
from numbers import Number
from typing import Union, Optional, List, Sequence

import cupy as cp

# local
import ivy
from ivy.functional.backends.cupy.device import _to_device
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
    dtype: Optional[cp.dtype] = None,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if dtype:
        dtype = as_native_dtype(dtype)
    res = _to_device(cp.arange(start, stop, step=step, dtype=dtype), device=device)
    if not dtype:
        if res.dtype == cp.float64:
            return res.astype(cp.float32)
        elif res.dtype == cp.int64:
            return res.astype(cp.int32)
    return res


@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
def asarray(
    obj: Union[cp.ndarray, bool, int, float, NestedSequence, SupportsBufferProtocol],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[cp.dtype] = None,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    with open("it_ran.txt", mode="w") as file:
        file.write("hi")

    # If copy=none then try using existing memory buffer
    if isinstance(obj, cp.ndarray) and dtype is None:
        dtype = obj.dtype
    elif isinstance(obj, (list, tuple, dict)) and len(obj) != 0 and dtype is None:
        dtype = ivy.default_dtype(item=obj, as_native=True)
        if copy is True:
            return _to_device(cp.copy(cp.asarray(obj, dtype=dtype)), device=device)
        else:
            return _to_device(cp.asarray(obj, dtype=dtype), device=device)
    else:
        dtype = ivy.default_dtype(dtype=dtype, item=obj)
    if copy is True:
        return _to_device(cp.copy(cp.asarray(obj, dtype=dtype)), device=device)
    else:
        return _to_device(cp.asarray(obj, dtype=dtype), device=device)


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: cp.dtype,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return _to_device(cp.empty(shape, dtype), device=device)


def empty_like(
    x: cp.ndarray, /, *, dtype: cp.dtype, device: str, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return _to_device(cp.empty_like(x, dtype=dtype), device=device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: cp.dtype,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if n_cols is None:
        n_cols = n_rows
    i = cp.eye(n_rows, n_cols, k, dtype)
    if batch_shape is None:
        return _to_device(i, device=device)
    else:
        reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
        tile_dims = list(batch_shape) + [1, 1]
        return_mat = cp.tile(cp.reshape(i, reshape_dims), tile_dims)
        return _to_device(return_mat, device=device)


def from_dlpack(x, /, *, out: Optional[cp.ndarray] = None):
    return cp.from_dlpack(x)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, cp.dtype]] = None,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    dtype = ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True)
    ivy.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
    return _to_device(
        cp.full(shape, fill_value, dtype),
        device=device,
    )


def full_like(
    x: cp.ndarray,
    /,
    fill_value: float,
    *,
    dtype: cp.dtype,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    ivy.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
    return _to_device(cp.full_like(x, fill_value, dtype=dtype), device=device)


def linspace(
    start: Union[cp.ndarray, float],
    stop: Union[cp.ndarray, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: cp.dtype,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if axis is None:
        axis = -1
    ans = cp.linspace(start, stop, num, endpoint, dtype=dtype, axis=axis)
    # Waiting for fix when start is -0.0: https://github.com/numpy/numpy/issues/21513
    if (
        ans.shape[0] >= 1
        and (not isinstance(start, cp.ndarray))
        and (not isinstance(stop, cp.ndarray))
    ):
        ans[0] = start
    return _to_device(ans, device=device)


def meshgrid(
    *arrays: cp.ndarray, sparse: bool = False, indexing: str = "xy"
) -> List[cp.ndarray]:
    return cp.meshgrid(*arrays, sparse=sparse, indexing=indexing)


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: cp.dtype,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return _to_device(cp.ones(shape, dtype), device=device)


def ones_like(
    x: cp.ndarray, /, *, dtype: cp.dtype, device: str, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return _to_device(cp.ones_like(x, dtype=dtype), device=device)


def tril(
    x: cp.ndarray, /, *, k: int = 0, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.tril(x, k)


def triu(
    x: cp.ndarray, /, *, k: int = 0, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.triu(x, k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: cp.dtype,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return _to_device(cp.zeros(shape, dtype), device=device)


def zeros_like(
    x: cp.ndarray, /, *, dtype: cp.dtype, device: str, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return _to_device(cp.zeros_like(x, dtype=dtype), device=device)


# Extra #
# -------#


array = asarray


def copy_array(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return x.copy()


def logspace(
    start: Union[cp.ndarray, int],
    stop: Union[cp.ndarray, int],
    /,
    num: int,
    *,
    base: float = 10.0,
    axis: Optional[int] = None,
    dtype: cp.dtype,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if axis is None:
        axis = -1
    return _to_device(
        cp.logspace(start, stop, num=num, base=base, dtype=dtype, axis=axis),
        device=device,
    )


def one_hot(
    indices: cp.ndarray,
    depth: int,
    /,
    *,
    on_value: Optional[Number] = None,
    off_value: Optional[Number] = None,
    axis: Optional[int] = None,
    dtype: Optional[cp.dtype] = None,
    device: str,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    on_none = on_value is None
    off_none = off_value is None

    if dtype is None:
        if on_none and off_none:
            dtype = cp.float32
        else:
            if not on_none:
                dtype = cp.array(on_value).dtype
            elif not off_none:
                dtype = cp.array(off_value).dtype

    res = cp.eye(depth, dtype=dtype)[cp.array(indices, dtype="int64").reshape(-1)]
    res = res.reshape(list(indices.shape) + [depth])

    if not on_none and not off_none:
        res = cp.where(res == 1, on_value, off_value)

    if axis is not None:
        res = cp.moveaxis(res, -1, axis)

    return res
