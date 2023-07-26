# global
import mxnet as mx
import numpy as np
from numbers import Number
from typing import Union, List, Optional, Sequence, Tuple

# lcoal
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
    asarray_inputs_to_native_shapes,
)


def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[None] = None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
@asarray_inputs_to_native_shapes
def asarray(
    obj: Union[
        (
            None,
            mx.ndarray.NDArray,
            bool,
            int,
            float,
            NestedSequence,
            SupportsBufferProtocol,
            np.ndarray,
        )
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[None] = None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    ret = mx.nd.array(obj, device, dtype=dtype)
    if copy:
        return mx.numpy.copy(ret)
    return ret


array = asarray


def empty(
    *size: Union[(int, Sequence[int])],
    shape: Optional[ivy.NativeShape] = None,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def empty_like(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[(int, Sequence[int])]] = None,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def from_dlpack(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def full(
    shape: Union[(ivy.NativeShape, Sequence[int])],
    fill_value: Union[(int, float, bool)],
    *,
    dtype: Optional[Union[(ivy.Dtype, None)]] = None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def full_like(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    fill_value: Number,
    *,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def linspace(
    start: Union[(None, mx.ndarray.NDArray, float)],
    stop: Union[(None, mx.ndarray.NDArray, float)],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
):
    raise IvyNotImplementedException()


def meshgrid(
    *arrays: Union[(None, mx.ndarray.NDArray)],
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()


def ones(
    shape: Optional[ivy.NativeShape] = None,
    *,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    return mx.nd.ones(shape, dtype=dtype, ctx=device)


def ones_like(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    return mx.nd.ones_like(x, dtype=dtype, ctx=device)


def tril(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    k: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def triu(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    k: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def zeros(
    *size: Union[(int, Sequence[int])],
    shape: Optional[ivy.NativeShape] = None,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def zeros_like(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    if x.shape == ():
        ret = mx.nd.array(0, dtype=dtype)
    else:
        ret = mx.ndarray.zeros_like(x, dtype=dtype)
    return ivy.to_device(ret, device)


def copy_array(
    x: Union[(None, mx.ndarray.NDArray)],
    *,
    to_ivy_array: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    if to_ivy_array:
        return ivy.to_ivy(x.copy())
    return x.copy()


def one_hot(
    indices: Union[(None, mx.ndarray.NDArray)],
    depth: int,
    /,
    *,
    on_value: Optional[Number] = 1,
    off_value: Optional[Number] = 0,
    axis: Optional[int] = -1,
    dtype: Optional[None] = None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def frombuffer(
    buffer: bytes,
    dtype: Optional[None] = float,
    count: Optional[int] = (-1),
    offset: Optional[int] = 0,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def triu_indices(
    n_rows: int, n_cols: Optional[int] = None, k: int = 0, /, *, device: str
) -> Tuple[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()
