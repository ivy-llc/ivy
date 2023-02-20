"""Collection of Paddle network layers, wrapped to fit Ivy syntax and signature."""

from typing import Optional, Tuple, Union, Sequence

# global
import paddle

# local
import ivy
from ivy.exceptions import IvyNotImplementedException
from . import backend_version


def conv1d(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


# noinspection PyUnresolvedReferences
def conv1d_transpose(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


# noinspection PyUnresolvedReferences
def conv2d(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


# noinspection PyUnresolvedReferences
def conv2d_transpose(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: Optional[str] = "NHWC",
    dilations: Optional[Union[int, Tuple[int, int]]] = 1,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


# noinspection PyUnresolvedReferences
def depthwise_conv2d(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: Optional[str] = "NHWC",
    dilations: Optional[Union[int, Tuple[int, int]]] = 1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


# noinspection PyUnresolvedReferences
def conv3d(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: Optional[str] = "NDHWC",
    dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


# noinspection PyUnresolvedReferences
def conv3d_transpose(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: Optional[str] = "NDHWC",
    dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def conv_general_dilated(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: Optional[int] = 2,
    data_format: Optional[str] = "channel_last",
    feature_group_count: Optional[int] = 1,
    x_dilations: Optional[
        Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
    ] = 1,
    dilations: Optional[
        Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
    ] = 1,
    bias: Optional[paddle.Tensor] = None,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def conv_general_transpose(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: Optional[int] = 2,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: Optional[str] = "NDHWC",
    dilations: Optional[
        Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
    ] = 1,
    feature_group_count: Optional[int] = 1,
    bias: Optional[paddle.Tensor] = None,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()
