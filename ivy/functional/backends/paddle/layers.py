"""Collection of Paddle network layers, wrapped to fit Ivy syntax and signature."""

from typing import Optional, Tuple, Union, Sequence

# global
import paddle
import ivy
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.functional.ivy.layers import (
    _handle_padding,
    _get_x_data_format,
    _deconv_length,
)
import ivy.functional.backends.paddle as paddle_backend

# local

from . import backend_version


def _is_list_or_tuple(inp):
    return isinstance(inp, (list, tuple))


def _convert_to_list(value, n, name="padding", _type=int):
    if isinstance(value, _type):
        return [value] * n
    else:
        try:
            value_list = list(value)
        except TypeError:
            raise ValueError(
                f"The input {name}'s type must be list or tuple. Received: {value}"
            )
        else:
            return value_list


def _pad_before_conv(x, filters, strides, padding, dims, dilations, data_format):
    dilations = _convert_to_list(dilations, dims, "dilations")
    strides = _convert_to_list(strides, dims, "strides")

    if isinstance(padding, str):
        # Case 1: "VALID", "SAME" etc.
        filter_shape = [
            filters.shape[i] + (filters.shape[i] - 1) * (dilations[i] - 1)
            for i in range(dims)
        ]
        padding_spec = [
            _handle_padding(x.shape[1 + i], strides[i], filter_shape[i], padding)
            for i in range(dims - 1, -1, -1)
        ]
        padding_top = [padding_spec[i] // 2 for i in range(dims)]
        padding_bot = [padding_spec[i] - padding_spec[i] // 2 for i in range(dims)]
        padding = [None] * len(padding_top) * 2
        padding[::2] = padding_top
        padding[1::2] = padding_bot

    else:
        if isinstance(padding, int):
            padding = [(padding, padding)] * dims
        if (
            _is_list_or_tuple(padding)
            and len(padding) == dims
            and _is_list_or_tuple(padding[0])
        ):
            # Case 2: [(pad_left, pad_right), (pad_top, pad_bottom)...]
            padding = [item for sublist in padding for item in sublist[::-1]][::-1]
        else:
            raise ValueError(f"Invalid padding format: {padding}")

    if not all([p >= 0 for p in padding]):
        raise ValueError(
            "Invalid padding, all values should be larger than"
            f"or equal to 0, but received: {padding}."
        )

    return paddle.nn.functional.pad(
        x, pad=padding, data_format=data_format, mode="constant"
    )


def _pad_before_conv_tranpose(
    x, filters, strides, padding, dims, dilations, output_shape, filter_shape
):
    if output_shape is None:
        out_shape = [
            _deconv_length(
                x.shape[i + 2], strides[i], filter_shape[i], padding, dilations[i]
            )
            for i in range(dims)
        ]
        output_shape = [x.shape[0], *out_shape, filters.shape[1]]
    elif len(output_shape) == dims:
        output_shape = [x.shape[0]] + output_shape + [filters.shape[1]]
    not_valid_pad = [False] * dims
    filter_shape = [
        filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
        for i in range(dims)
    ]
    pad_specific = [
        _handle_padding(output_shape[i + 1], strides[i], filter_shape[i], padding)
        for i in range(dims)
    ]
    if padding == "VALID":
        padding_list = [0] * dims
    else:
        for i in range(dims):
            if pad_specific[i] % 2 != 0:
                pad_specific[i] -= 1
                not_valid_pad[i] = True
        padding_list = [pad_specific[i] // 2 for i in range(dims)]
    out_shape = [
        (x.shape[i + 2] - 1) * strides[i]
        - 2 * padding_list[i]
        + dilations[i] * (filters.shape[i + 2] - 1)
        + 1
        for i in range(dims)
    ]
    output_padding = [max(output_shape[i + 1] - out_shape[i], 0) for i in range(dims)]
    return not_valid_pad, padding_list, output_padding


def conv1d(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
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
    if data_format == "NCW":
        x = tf.transpose(x, (0, 2, 1))
        filters = tf.transpose(filters, (0, 2, 1))
        output_shape = _output_shape(
        x.shape, filters.shape, output_shape, strides, padding, 1, dilations
    )
    res = tf.nn.conv1d_transpose(
        x, filters, output_shape, strides, padding, "NWC", dilations
    )
    if data_format == "NCW":
        res = tf.transpose(res, (0, 2, 1))
    return res
    #raise IvyNotImplementedException()


# noinspection PyUnresolvedReferences
def conv2d(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("float16",)}},
    backend_version,
)
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
    if data_format == "NHWC":
        x = x.transpose([0, 3, 1, 2])
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    filters = filters.transpose([2, 3, 0, 1])
    not_valid_pad, padding_list, output_padding = _pad_before_conv_tranpose(
        x, filters, strides, padding, 2, dilations, output_shape, filters.shape[2:]
    )
    res = paddle.nn.functional.conv2d_transpose(
        x,
        filters,
        stride=strides,
        padding=padding_list,
        output_padding=output_padding,
        dilation=dilations,
        data_format="NCHW",
    )
    if not_valid_pad[0]:
        res = res[:, :, 0:-1, :]
    if not_valid_pad[1]:
        res = res[:, :, :, 0:-1]
    if data_format == "NHWC":
        res = res.transpose([0, 2, 3, 1])
    return res


# noinspection PyUnresolvedReferences
def depthwise_conv2d(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: Optional[str] = "NHWC",
    dilations: Optional[Union[int, Tuple[int, int]]] = 1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("float16",)}},
    backend_version,
)
def conv3d(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: Optional[str] = "NDHWC",
    dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
    out: Optional[paddle.Tensor] = None,
):
    if data_format == "NCDHW":
        x = paddle.transpose(x, perm=(0, 2, 3, 4, 1))

    df = "NDHWC"
    x = _pad_before_conv(x, filters, strides, padding, 3, dilations, df)
    filters = paddle.transpose(filters, perm=(4, 3, 0, 1, 2))
    padding = "VALID"

    res = paddle.nn.functional.conv3d(
        x,
        filters,
        data_format=df,
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


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("float16",)}},
    backend_version,
)
def conv_general_dilated(
    x: paddle.Tensor,
    filters: paddle.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: Optional[int] = 2,
    data_format: Optional[str] = "channel_last",
    filter_format: Optional[str] = "channel_last",
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
    if data_format == "channel_first":
        x = paddle.transpose(x, perm=(0, *range(2, dims + 2), 1))

    if filter_format == "channel_first":
        filters = paddle.transpose(filters, (*range(2, dims + 2), 1, 0))

    # adding dilation in input
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    for i in range(dims):
        if x_dilations[i] > 1:
            h = x.shape[1 + i]
            new_height = h + (h - 1) * (x_dilations[i] - 1)
            h = paddle.eye(new_height, dtype=x.dtype)[:: x_dilations[i]]
            x = paddle_backend.swapaxes(x, 1 + i, -1)
            x = paddle.matmul(x, h)
            x = paddle_backend.swapaxes(x, -1, 1 + i)

    df = "NLC" if dims == 1 else _get_x_data_format(dims, data_format="channel_last")
    x = _pad_before_conv(x, filters, strides, padding, dims, dilations, df)
    filters = paddle.transpose(filters, perm=(dims + 1, dims, *range(dims)))
    padding = "VALID"

    if dims == 1:
        res = paddle.nn.functional.conv1d(
            x,
            filters,
            bias=bias,
            data_format=df,
            stride=strides,
            padding=padding,
            dilation=dilations,
            groups=feature_group_count,
        )
    elif dims == 2:
        res = paddle.nn.functional.conv2d(
            x,
            filters,
            bias=bias,
            data_format=df,
            stride=strides,
            padding=padding,
            dilation=dilations,
            groups=feature_group_count,
        )
    elif dims == 3:
        res = paddle.nn.functional.conv3d(
            x,
            filters,
            bias=bias,
            data_format=df,
            stride=strides,
            padding=padding,
            dilation=dilations,
            groups=feature_group_count,
        )

    if data_format == "channel_first":
        res = paddle.transpose(res, perm=(0, dims + 1, *range(1, dims + 1)))
    return res


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
