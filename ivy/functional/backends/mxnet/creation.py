import mxnet as mx
import numpy as np
from numbers import Number
from typing import Union, List, Optional, Sequence
import ivy
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
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
    raise NotImplementedError("mxnet.arange Not Implemented")


@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
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


def empty(
    *size: Union[(int, Sequence[int])],
    shape: Optional[ivy.NativeShape] = None,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.empty Not Implemented")


def empty_like(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.empty_like Not Implemented")


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
    raise NotImplementedError("mxnet.eye Not Implemented")


def from_dlpack(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.from_dlpack Not Implemented")


def full(
    shape: Union[(ivy.NativeShape, Sequence[int])],
    fill_value: Union[(int, float, bool)],
    *,
    dtype: Optional[Union[(ivy.Dtype, None)]] = None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.full Not Implemented")


def full_like(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    fill_value: Number,
    *,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.full_like Not Implemented")


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
    raise NotImplementedError("mxnet.linspace Not Implemented")


def meshgrid(
    *arrays: Union[(None, mx.ndarray.NDArray)],
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.meshgrid Not Implemented")


def ones(
    *size: Union[(int, Sequence[int])],
    shape: Optional[ivy.NativeShape] = None,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.ones Not Implemented")


def ones_like(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.ones_like Not Implemented")


def tril(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    k: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.tril Not Implemented")


def triu(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    k: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.triu Not Implemented")


def zeros(
    *size: Union[(int, Sequence[int])],
    shape: Optional[ivy.NativeShape] = None,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.zeros Not Implemented")


def zeros_like(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    dtype: None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.zeros_like Not Implemented")


array = asarray


def copy_array(
    x: Union[(None, mx.ndarray.NDArray)],
    *,
    to_ivy_array: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.copy_array Not Implemented")


def one_hot(
    indices: Union[(None, mx.ndarray.NDArray)],
    depth: int,
    /,
    *,
    on_value: Optional[Number] = None,
    off_value: Optional[Number] = None,
    axis: Optional[int] = None,
    dtype: Optional[None] = None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.one_hot Not Implemented")
