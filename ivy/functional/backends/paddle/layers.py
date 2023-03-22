"""Collection of Paddle network layers, wrapped to fit Ivy syntax and signature."""

from typing import Optional, Tuple, Union, Sequence

# global
import paddle

# local
import ivy
from ivy.utils.exceptions import IvyNotImplementedException

# from . import backend_version
from ivy.functional.ivy.layers import _handle_padding


def _is_list_or_tuple(inp):
    return isinstance(inp, (list, tuple))


def _pad_before_conv(x, filters, strides, padding, dims, dilations, data_format):
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    strides = [strides] * dims if isinstance(strides, int) else strides
    if isinstance(padding, str):
        # Case 1: "VALID", "SAME" etc.
        filter_shape = [
            filters.shape[i] + (filters.shape[i] - 1) * (dilations[i] - 1)
            for i in range(dims)
        ]
        padding_spec = [
            _handle_padding(x.shape[1 + i], strides[i], filter_shape[i], padding)
            for i in range(dims)
        ]
        padding_top = [padding_spec[i] // 2 for i in range(dims)]
        padding_bot = [padding_spec[i] - padding_spec[i] // 2 for i in range(dims)]
        padding = [None] * len(padding_top) * 2
        padding[::2] = padding_top
        padding[1::2] = padding_bot
    elif _is_list_or_tuple(padding):
        if len(padding) == dims + 2 and _is_list_or_tuple(padding[0]):
            # Case 2: [(0,0),(pad_left, pad_right),(pad_top, pad_bottom)...,(0,0)]
            padding = padding[1:-1] if data_format == "NDHWC" else padding[2:]
            padding = [elem for pad_i_dim in padding for elem in pad_i_dim]
        elif len(padding) == dims and _is_list_or_tuple(padding[0]):
            # Case 3: [(pad_left, pad_right), (pad_top, pad_bottom)...]
            padding = [elem for pad_i_dim in padding for elem in pad_i_dim]
        else:
            raise ValueError(f"Invalid padding format: {padding}")
    return paddle.nn.functional.pad(
        x, data_format="NDHWC", mode="constant", pad=padding
    )


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
    if data_format == "NCDHW":
        x = paddle.transpose(x, perm=(0, 2, 3, 4, 1))
    x = _pad_before_conv(x, filters, strides, padding, 3, dilations, data_format)
    filters = paddle.transpose(filters, perm=(4, 3, 0, 1, 2))
    if not isinstance(padding, str):
        padding = "VALID"
    res = paddle.nn.functional.conv3d(
        x,
        filters,
        data_format="NDHWC",
        stride=strides,
        padding=padding,
        dilation=dilations,
    )
    if data_format == "NCDHW":
        res = paddle.transpose(res, perm=(0, 4, 1, 2, 3))
    return res


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
