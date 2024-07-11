"""Collection of MXNet network layers, wrapped to fit Ivy syntax and
signature."""

# global
import mxnet as mx
from typing import Optional, Tuple, Union, Sequence
import ivy

# local
from ivy.utils.exceptions import IvyNotImplementedException


def conv1d(
    x: Union[(None, mx.ndarray.NDArray)],
    filters: Union[(None, mx.ndarray.NDArray)],
    strides: Union[(int, Tuple[int])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    data_format: str = "NWC",
    dilations: Union[(int, Tuple[int])] = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def conv1d_transpose(
    x: Union[(None, mx.ndarray.NDArray)],
    filters: Union[(None, mx.ndarray.NDArray)],
    strides: Union[(int, Tuple[int])],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NWC",
    dilations: Union[(int, Tuple[int])] = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
):
    raise IvyNotImplementedException()


def conv2d(
    x: Union[(None, mx.ndarray.NDArray)],
    filters: Union[(None, mx.ndarray.NDArray)],
    strides: Union[(int, Tuple[(int, int)])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[(int, Tuple[(int, int)])] = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def conv2d_transpose(
    x: Union[(None, mx.ndarray.NDArray)],
    filters: Union[(None, mx.ndarray.NDArray)],
    strides: Union[(int, Tuple[(int, int)])],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NHWC",
    dilations: Union[(int, Tuple[(int, int)])] = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
):
    raise IvyNotImplementedException()


def depthwise_conv2d(
    x: Union[(None, mx.ndarray.NDArray)],
    filters: Union[(None, mx.ndarray.NDArray)],
    strides: Union[(int, Tuple[(int, int)])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[(int, Tuple[(int, int)])] = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def conv3d(
    x: Union[(None, mx.ndarray.NDArray)],
    filters: Union[(None, mx.ndarray.NDArray)],
    strides: Union[(int, Tuple[(int, int, int)])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    data_format: str = "NDHWC",
    dilations: Union[(int, Tuple[(int, int, int)])] = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
):
    raise IvyNotImplementedException()


def conv3d_transpose(
    x: None,
    filters: None,
    strides: Union[(int, Tuple[(int, int, int)])],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NDHWC",
    dilations: Union[(int, Tuple[(int, int, int)])] = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> None:
    raise IvyNotImplementedException()


def conv_general_dilated(
    x: Union[(None, mx.ndarray.NDArray)],
    filters: Union[(None, mx.ndarray.NDArray)],
    strides: Union[(int, Tuple[int], Tuple[(int, int)], Tuple[(int, int, int)])],
    padding: Union[(str, int, Sequence[Tuple[(int, int)]])],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[
        (int, Tuple[int], Tuple[(int, int)], Tuple[(int, int, int)])
    ] = 1,
    dilations: Union[(int, Tuple[int], Tuple[(int, int)], Tuple[(int, int, int)])] = 1,
    bias: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def conv_general_transpose(
    x: Union[(None, mx.ndarray.NDArray)],
    filters: Union[(None, mx.ndarray.NDArray)],
    strides: Union[(int, Tuple[(int, int)])],
    padding: str,
    /,
    *,
    dims: int = 2,
    filter_format: str = "channel_last",
    data_format: str = "channel_last",
    output_shape: Optional[Union[(ivy.NativeShape, Sequence[int])]] = None,
    dilations: Union[(int, Tuple[int], Tuple[(int, int)], Tuple[(int, int, int)])] = 1,
    feature_group_count: int = 1,
    bias: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()
