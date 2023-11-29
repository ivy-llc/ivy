"""Collection of TensorFlow network layers, wrapped to fit Ivy syntax and signature."""

# global
from typing import Optional, Tuple, Union, Sequence

import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
from . import backend_version
from ivy.functional.ivy.layers import (
    _deconv_length,
    _get_x_data_format,
)


@with_supported_dtypes(
    {"2.15.0 and below": ("float", "int32", "int64", "complex")},
    backend_version,
)
def linear(
    x: Union[tf.Tensor, tf.Variable],
    weight: Union[tf.Tensor, tf.Variable],
    /,
    *,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.einsum("...i,...ji->...j", x, weight) + bias


def _x_dil_before_conv(x, dims, x_dilations, data_format):
    # adding dilation in input
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    x_dilations_idxs = [i for i, x_dil in enumerate(x_dilations) if x_dil > 1]
    if x_dilations_idxs:
        if data_format[-1] == "C":
            offset = 1
        else:
            offset = 2
        for i in x_dilations_idxs:
            h = x.shape[offset + i]
            new_height = h + (h - 1) * (x_dilations[i] - 1)
            h = tf.eye(new_height, dtype=x.dtype)[:: x_dilations[i]]
            x = tf.experimental.numpy.swapaxes(x, offset + i, -1)
            x = tf.matmul(x, h)
            x = tf.experimental.numpy.swapaxes(x, -1, offset + i)
    return x


def _pad_before_conv(x, padding, dims, data_format):
    if isinstance(padding, str):
        return x, padding
    elif isinstance(padding, int):
        pad_list = [(padding, padding)] * dims
    else:
        pad_list = padding
    if data_format[-1] == "C":
        pad_list = [(0, 0), *pad_list, (0, 0)]
    else:
        pad_list = [(0, 0), (0, 0), *pad_list]
    return tf.pad(x, pad_list, "CONSTANT"), "VALID"


def _extend_2d_padding(padding, data_format):
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        padding = [(padding, padding)] * 2
    if data_format[-1] == "C":
        padding = [(0, 0)] + padding + [(0, 0)]
    else:
        padding = [(0, 0), (0, 0)] + padding
    return padding


def _transpose_out_pad(
    x_shape, filter_shape, strides, padding, dims, dilations, data_format
):
    if data_format[-1] == "C":
        offset = 1
    else:
        offset = 2
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    strides = [strides] * dims if isinstance(strides, int) else strides
    if isinstance(padding, str):
        out_shape = [
            _deconv_length(
                x_shape[offset + i], strides[i], filter_shape[i], padding, dilations[i]
            )
            for i in range(dims)
        ]
    else:
        if isinstance(padding, int):
            padding = [[padding, padding]] * dims
        out_shape = [
            (x_shape[offset + i] - 1) * strides[i]
            - padding[i][0]
            - padding[i][1]
            + dilations[i] * (filter_shape[i] - 1)
            + 1
            for i in range(dims)
        ]
        if data_format[-1] == "C":
            padding = [[0, 0], *padding, [0, 0]]
        else:
            padding = [[0, 0], [0, 0], *padding]
    if data_format[-1] == "C":
        out_shape = [x_shape[0], *out_shape, filter_shape[-2]]
    else:
        out_shape = [x_shape[0], filter_shape[-2], *out_shape]
    return out_shape, padding


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def conv1d(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int]] = 1,
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    permuted_x = False
    if data_format == "NCW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 1))
        data_format = "NWC"
        permuted_x = True
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (2, 1, 0))
    x = _x_dil_before_conv(x, 1, x_dilations, data_format)
    x, padding = _pad_before_conv(x, padding, 1, data_format)
    res = tf.nn.conv1d(x, filters, strides, padding, data_format, dilations)
    if bias is not None:
        if data_format[1] == "C":
            bias = bias[tf.newaxis, ..., tf.newaxis]
        res = tf.math.add(res, bias)
    if permuted_x:
        res = tf.transpose(res, (0, 2, 1))
    return res


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def conv1d_transpose(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    if ivy.dev(x) == "cpu" and (
        (dilations > 1) if isinstance(dilations, int) else any(d > 1 for d in dilations)
    ):
        raise ivy.utils.exceptions.IvyException(
            "Tensorflow does not support dilations greater than 1 when device is cpu"
        )
    permuted_x = False
    if data_format == "NCW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 1))
        data_format = "NWC"
        permuted_x = True
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (2, 1, 0))
    output_shape, padding = _transpose_out_pad(
        x.shape, filters.shape, strides, padding, 1, dilations, data_format
    )
    res = tf.nn.conv1d_transpose(
        x, filters, output_shape, strides, padding, data_format, dilations
    )
    if bias is not None:
        if data_format[1] == "C":
            bias = bias[tf.newaxis, ..., tf.newaxis]
        res = tf.math.add(res, bias)
    if permuted_x:
        res = tf.transpose(res, (0, 2, 1))
    return res


def _extend_3d_strides_dilations(strides, dilations, data_format):
    if data_format[-1] == "C":
        strides = [1, *([strides] * 3 if isinstance(strides, int) else strides), 1]
        dilations = [
            1,
            *([dilations] * 3 if isinstance(dilations, int) else dilations),
            1,
        ]
    else:
        strides = [1, 1, *([strides] * 3 if isinstance(strides, int) else strides)]
        dilations = [
            1,
            1,
            *([dilations] * 3 if isinstance(dilations, int) else dilations),
        ]
    return strides, dilations


@with_supported_dtypes({"2.14.0 and below": ("float", "int32")}, backend_version)
def conv2d(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    permuted_x = False
    if data_format == "NCHW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 3, 1))
        data_format = "NHWC"
        permuted_x = True
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (2, 3, 1, 0))
    x = _x_dil_before_conv(x, 2, x_dilations, data_format)
    padding = _extend_2d_padding(padding, data_format)
    res = tf.nn.conv2d(x, filters, strides, padding, data_format, dilations)
    if bias is not None:
        if data_format[1] == "C":
            bias = bias[tf.newaxis, ..., tf.newaxis, tf.newaxis]
        res = tf.math.add(res, bias)
    if permuted_x:
        return tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def conv2d_transpose(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    if ivy.dev(x) == "cpu" and (
        (dilations > 1) if isinstance(dilations, int) else any(d > 1 for d in dilations)
    ):
        raise ivy.utils.exceptions.IvyException(
            "Tensorflow does not support dilations greater than 1 when device is cpu"
        )
    permuted_x = False
    if data_format == "NCHW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 3, 1))
        data_format = "NHWC"
        permuted_x = True
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (2, 3, 1, 0))
    output_shape, padding = _transpose_out_pad(
        x.shape,
        filters.shape,
        strides,
        padding,
        2,
        dilations,
        data_format,
    )
    res = tf.nn.conv2d_transpose(
        x, filters, output_shape, strides, padding, data_format, dilations
    )
    if bias is not None:
        if data_format[1] == "C":
            bias = bias[tf.newaxis, ..., tf.newaxis, tf.newaxis]
        res = tf.math.add(res, bias)
    if permuted_x:
        return tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def depthwise_conv2d(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    permuted_x = False
    if data_format == "NCHW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 3, 1))
        data_format = "NHWC"
        permuted_x = True
    if tf.rank(filters) == 3:
        filters = tf.expand_dims(filters, -1)
    padding = _extend_2d_padding(padding, data_format)
    strides = [1, strides[0], strides[1], 1]
    res = tf.nn.depthwise_conv2d(x, filters, strides, padding, data_format, dilations)
    if permuted_x:
        res = tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def conv3d(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    permuted_x = False
    if data_format == "NCDHW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
        data_format = "NDHWC"
        permuted_x = True
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (2, 3, 4, 1, 0))
    x = _x_dil_before_conv(x, 3, x_dilations, data_format)
    x, padding = _pad_before_conv(x, padding, 3, data_format)
    strides, dilations = _extend_3d_strides_dilations(strides, dilations, data_format)
    res = tf.nn.conv3d(x, filters, strides, padding, data_format, dilations)
    if bias is not None:
        if data_format[1] == "C":
            bias = bias[tf.newaxis, ..., tf.newaxis, tf.newaxis, tf.newaxis]
        res = tf.math.add(res, bias)
    if permuted_x:
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def conv3d_transpose(
    x: Tensor,
    filters: Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tensor:
    if ivy.dev(x) == "cpu" and (
        (dilations > 1) if isinstance(dilations, int) else any(d > 1 for d in dilations)
    ):
        raise ivy.utils.exceptions.IvyException(
            "Tensorflow does not support dilations greater than 1 when device is cpu"
        )
    permuted_x = False
    if data_format == "NCDHW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
        data_format = "NDHWC"
        permuted_x = True
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (2, 3, 4, 1, 0))
    output_shape, padding = _transpose_out_pad(
        x.shape, filters.shape, strides, padding, 3, dilations, data_format
    )
    strides, dilations = _extend_3d_strides_dilations(strides, dilations, data_format)
    res = tf.nn.conv3d_transpose(
        x, filters, output_shape, strides, padding, data_format, dilations
    )
    if bias is not None:
        if data_format[1] == "C":
            bias = bias[tf.newaxis, ..., tf.newaxis, tf.newaxis, tf.newaxis]
        res = tf.math.add(res, bias)
    if permuted_x:
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def conv_general_dilated(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    filter_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (*range(2, dims + 2), 1, 0))

    num_channels = x.shape[1] if data_format == "channel_first" else x.shape[-1]
    if filters.shape[-2] != (num_channels // feature_group_count):
        raise ivy.utils.exceptions.IvyError(
            f"given feature_group_count {feature_group_count} expected input channel of"
            f" the filter to be {num_channels // feature_group_count} but got"
            f" {filters.shape[-2]}"
        )
    if num_channels % feature_group_count != 0:
        raise ivy.utils.exceptions.IvyError(
            "input channel should be divisible by feature group count"
            f" {feature_group_count} but got input channel {num_channels}"
        )

    permuted_x = False
    if data_format == "channel_first" and (
        ivy.dev(x) == "cpu" or feature_group_count != 1
    ):
        x = tf.transpose(x, (0, *range(2, dims + 2), 1))
        data_format = "channel_last"
        permuted_x = True

    data_format = _get_x_data_format(dims, data_format)

    x = _x_dil_before_conv(x, dims, x_dilations, data_format)

    if dims == 2:
        padding = _extend_2d_padding(padding, data_format)
        if feature_group_count == 1:
            res = tf.nn.conv2d(
                x,
                filters,
                strides,
                padding,
                data_format=data_format,
                dilations=dilations,
            )
        else:
            res = tf.concat(
                [
                    tf.nn.conv2d(
                        x[..., i : i + filters.shape[-2]],
                        filters[..., j : j + filters.shape[-1] // feature_group_count],
                        strides,
                        padding,
                        data_format,
                        dilations,
                    )
                    for i, j in zip(
                        range(0, x.shape[-1], filters.shape[-2]),
                        range(
                            0,
                            filters.shape[-1],
                            filters.shape[-1] // feature_group_count,
                        ),
                    )
                ],
                axis=-1,
            )
    else:
        x, padding = _pad_before_conv(x, padding, dims, data_format)
        if dims == 1:
            if feature_group_count == 1:
                res = tf.nn.conv1d(
                    x,
                    filters,
                    strides,
                    padding,
                    data_format=data_format,
                    dilations=dilations,
                )
            else:
                res = tf.concat(
                    [
                        tf.nn.conv1d(
                            x[..., i : i + filters.shape[-2]],
                            filters[
                                ..., j : j + filters.shape[-1] // feature_group_count
                            ],
                            strides,
                            padding,
                            data_format,
                            dilations,
                        )
                        for i, j in zip(
                            range(0, x.shape[-1], filters.shape[-2]),
                            range(
                                0,
                                filters.shape[-1],
                                filters.shape[-1] // feature_group_count,
                            ),
                        )
                    ],
                    axis=-1,
                )
        else:
            strides, dilations = _extend_3d_strides_dilations(
                strides, dilations, data_format
            )
            if feature_group_count == 1:
                res = tf.nn.conv3d(
                    x,
                    filters,
                    strides,
                    padding,
                    data_format=data_format,
                    dilations=dilations,
                )
            else:
                res = tf.concat(
                    [
                        tf.nn.conv3d(
                            x[..., i : i + filters.shape[-2]],
                            filters[
                                ..., j : j + filters.shape[-1] // feature_group_count
                            ],
                            strides,
                            padding,
                            data_format,
                            dilations,
                        )
                        for i, j in zip(
                            range(0, x.shape[-1], filters.shape[-2]),
                            range(
                                0,
                                filters.shape[-1],
                                filters.shape[-1] // feature_group_count,
                            ),
                        )
                    ],
                    axis=-1,
                )
    if bias is not None:
        if data_format[1] == "C":
            new_dims = [tf.newaxis, Ellipsis] + [tf.newaxis] * dims
            bias = bias[new_dims]
        res = tf.math.add(res, bias)
    if permuted_x:
        return tf.transpose(res, (0, dims + 1, *range(1, dims + 1)))
    return res


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def conv_general_transpose(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    dims: int = 2,
    filter_format: str = "channel_last",
    data_format: str = "channel_last",
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if feature_group_count == 1:
        if dims == 1:
            return conv1d_transpose(
                x,
                filters,
                strides,
                padding,
                output_shape=output_shape,
                filter_format=filter_format,
                data_format="NWC" if data_format == "channel_last" else "NCW",
                dilations=dilations,
                bias=bias,
            )
        elif dims == 2:
            return conv2d_transpose(
                x,
                filters,
                strides,
                padding,
                output_shape=output_shape,
                filter_format=filter_format,
                data_format="NHWC" if data_format == "channel_last" else "NCHW",
                dilations=dilations,
                bias=bias,
            )
        else:
            return conv3d_transpose(
                x,
                filters,
                strides,
                padding,
                output_shape=output_shape,
                filter_format=filter_format,
                data_format="NDHWC" if data_format == "channel_last" else "NCDHW",
                dilations=dilations,
                bias=bias,
            )
    else:
        if filter_format == "channel_first":
            filters = tf.transpose(filters, (*range(2, dims + 2), 1, 0))
        permuted_x = False
        if data_format == "channel_first" and ivy.dev(x) == "cpu":
            x = tf.transpose(x, (0, *range(2, dims + 2), 1))
            data_format = "channel_last"
            permuted_x = True

        data_format = _get_x_data_format(dims, data_format)

        output_shape, padding = _transpose_out_pad(
            x.shape, filters.shape, strides, padding, dims, dilations, data_format
        )

        if dims == 1:
            res = tf.concat(
                [
                    tf.nn.conv1d_transpose(
                        x[..., j : j + filters.shape[-2] // feature_group_count],
                        filters[
                            ..., j : j + filters.shape[-2] // feature_group_count, :
                        ],
                        output_shape,
                        strides,
                        padding=padding,
                        data_format=data_format,
                        dilations=dilations,
                    )
                    for j in range(
                        0, filters.shape[-2], filters.shape[-2] // feature_group_count
                    )
                ],
                axis=-1,
            )
        elif dims == 2:
            res = tf.concat(
                [
                    tf.nn.conv2d_transpose(
                        x[..., j : j + filters.shape[-2] // feature_group_count],
                        filters[
                            ..., j : j + filters.shape[-2] // feature_group_count, :
                        ],
                        output_shape,
                        strides,
                        padding=padding,
                        data_format=data_format,
                        dilations=dilations,
                    )
                    for j in range(
                        0, filters.shape[-2], filters.shape[-2] // feature_group_count
                    )
                ],
                axis=-1,
            )
        else:
            strides, dilations = _extend_3d_strides_dilations(
                strides, dilations, data_format
            )
            res = tf.concat(
                [
                    tf.nn.conv3d_transpose(
                        x[..., j : j + filters.shape[-2] // feature_group_count],
                        filters[
                            ..., j : j + filters.shape[-2] // feature_group_count, :
                        ],
                        output_shape,
                        strides,
                        padding=padding,
                        data_format=data_format,
                        dilations=dilations,
                    )
                    for j in range(
                        0, filters.shape[-2], filters.shape[-2] // feature_group_count
                    )
                ],
                axis=-1,
            )

        res = tf.math.add(res, bias) if bias is not None else res
        if permuted_x:
            return tf.transpose(res, (0, dims + 1, *range(1, dims + 1)))
        return res


def nms(
    boxes,
    scores=None,
    iou_threshold=0.5,
    max_output_size=None,
    score_threshold=float("-inf"),
):
    if scores is None:
        scores = tf.ones(boxes.shape[0])

    boxes = tf.gather(boxes, [1, 0, 3, 2], axis=1)
    ret = tf.image.non_max_suppression(
        boxes, scores, max_output_size or len(boxes), iou_threshold, score_threshold
    )

    return tf.cast(ret, dtype=tf.int64)
