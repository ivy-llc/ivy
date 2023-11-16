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


def _x_dil_before_conv(x, dims, x_dilations):
    # adding dilation in input
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    x_dilations_idxs = [i for i, x_dil in enumerate(x_dilations) if x_dil > 1]
    if x_dilations_idxs:
        for i in x_dilations_idxs:
            h = x.shape[1 + i]
            new_height = h + (h - 1) * (x_dilations[i] - 1)
            h = tf.eye(new_height, dtype=x.dtype)[:: x_dilations[i]]
            x = tf.experimental.numpy.swapaxes(x, 1 + i, -1)
            x = tf.matmul(x, h)
            x = tf.experimental.numpy.swapaxes(x, -1, 1 + i)
    return x


def _pad_before_conv(x, padding, dims):
    if isinstance(padding, str):
        return x, padding
    elif isinstance(padding, int):
        pad_list = [(padding, padding)] * dims
    else:
        pad_list = padding
    return (
        tf.pad(
            x,
            [
                (0, 0),
                *pad_list,
                (0, 0),
            ],
            "CONSTANT",
        ),
        "VALID",
    )


def _to_explicit_padding(padding, dims):
    if isinstance(padding, str):
        return padding, []
    if isinstance(padding, int):
        explicit_pad = [padding] * dims * 2
    else:
        explicit_pad = [item for sublist in padding for item in sublist]
    explicit_pad = [0, 0] + explicit_pad + [0, 0]
    return "EXPLICIT", explicit_pad


def _output_shape(
    x_shape, filter_shape, output_shape, strides, padding, dims, dilations
):
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    strides = [strides] * dims if isinstance(strides, int) else strides
    if output_shape is None:
        out_shape = [
            _deconv_length(
                x_shape[i + 1], strides[i], filter_shape[i], padding, dilations[i]
            )
            for i in range(dims)
        ]
        output_shape = [x_shape[0], *out_shape, filter_shape[-2]]
    elif len(output_shape) == dims:
        output_shape = [x_shape[0]] + output_shape + [filter_shape[-2]]
    return output_shape


@with_unsupported_dtypes({"2.14.0 and below": ("bfloat16", "complex")}, backend_version)
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
    if data_format == "NCW":
        x = tf.transpose(x, (0, 2, 1))
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (2, 1, 0))
    x = _x_dil_before_conv(x, 1, x_dilations)
    x, padding = _pad_before_conv(x, padding, 1)
    res = tf.nn.conv1d(x, filters, strides, padding, "NWC", dilations)
    res = tf.math.add(res, bias) if bias is not None else res
    if data_format == "NCW":
        res = tf.transpose(res, (0, 2, 1))
    return res


@with_unsupported_dtypes({"2.14.0 and below": ("bfloat16", "complex")}, backend_version)
def conv1d_transpose(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
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
    if data_format == "NCW":
        x = tf.transpose(x, (0, 2, 1))
    filters = tf.transpose(filters, (0, 2, 1))

    output_shape = _output_shape(
        x.shape, filters.shape, output_shape, strides, padding, 1, dilations
    )
    res = tf.nn.conv1d_transpose(
        x, filters, output_shape, strides, padding, "NWC", dilations
    )
    res = tf.math.add(res, bias) if bias is not None else res
    if data_format == "NCW":
        res = tf.transpose(res, (0, 2, 1))
    return res


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
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (2, 3, 1, 0))
    x = _x_dil_before_conv(x, 2, x_dilations)
    padding, explicit_padding = _to_explicit_padding(padding, 2)
    strides = [1] + list([strides] * 2 if isinstance(strides, int) else strides) + [1]
    dilations = (
        [1] + list([dilations] * 2 if isinstance(dilations, int) else dilations) + [1]
    )
    res = tf.raw_ops.Conv2D(
        input=x,
        filter=filters,
        strides=strides,
        padding=padding,
        explicit_paddings=explicit_padding,
        data_format="NHWC",
        dilations=dilations,
    )
    res = tf.math.add(res, bias) if bias is not None else res
    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes({"2.14.0 and below": ("bfloat16", "complex")}, backend_version)
def conv2d_transpose(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
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
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    filters = tf.transpose(filters, (0, 1, 3, 2))
    output_shape = _output_shape(
        x.shape, filters.shape, output_shape, strides, padding, 2, dilations
    )
    res = tf.nn.conv2d_transpose(
        x, filters, output_shape, strides, padding, "NHWC", dilations
    )
    res = tf.math.add(res, bias) if bias is not None else res
    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes({"2.14.0 and below": ("bfloat16", "complex")}, backend_version)
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
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    if tf.rank(filters) == 3:
        filters = tf.expand_dims(filters, -1)
    x, padding = _pad_before_conv(x, padding, 2)
    strides = [1, strides[0], strides[1], 1]
    res = tf.nn.depthwise_conv2d(x, filters, strides, padding, "NHWC", dilations)
    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes({"2.14.0 and below": ("bfloat16", "complex")}, backend_version)
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
    if data_format == "NCDHW":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    if filter_format == "channel_first":
        filters = tf.transpose(filters, (2, 3, 4, 1, 0))
    x = _x_dil_before_conv(x, 3, x_dilations)
    x, padding = _pad_before_conv(x, padding, 3)
    strides = [1] + list([strides] * 3 if isinstance(strides, int) else strides) + [1]
    dilations = (
        [1] + list([dilations] * 3 if isinstance(dilations, int) else dilations) + [1]
    )
    res = tf.nn.conv3d(x, filters, strides, padding, "NDHWC", dilations)
    res = tf.math.add(res, bias) if bias is not None else res
    if data_format == "NCDHW":
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res


@with_unsupported_dtypes({"2.14.0 and below": ("bfloat16", "complex")}, backend_version)
def conv3d_transpose(
    x: Tensor,
    filters: Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
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
    strides = [1] + list([strides] * 3 if isinstance(strides, int) else strides) + [1]
    dilations = (
        [1] + list([dilations] * 3 if isinstance(dilations, int) else dilations) + [1]
    )
    if data_format == "NCDHW":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    filters = tf.transpose(filters, (0, 1, 2, 4, 3))
    output_shape = _output_shape(
        x.shape, filters.shape, output_shape, strides[1:], padding, 3, dilations
    )
    res = tf.nn.conv3d_transpose(
        x, filters, output_shape, strides, padding, "NDHWC", dilations
    )
    res = tf.math.add(res, bias) if bias is not None else res
    if data_format == "NCDHW":
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res


@with_unsupported_dtypes({"2.14.0 and below": ("bfloat16", "complex")}, backend_version)
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
    # permuting dims based on formats
    if data_format == "channel_first":
        x = tf.transpose(x, (0, *range(2, dims + 2), 1))

    if filter_format == "channel_first":
        filters = tf.transpose(filters, (*range(2, dims + 2), 1, 0))

    x = _x_dil_before_conv(x, dims, x_dilations)

    df = _get_x_data_format(dims, "channel_last")

    if filters.shape[-2] != (x.shape[-1] // feature_group_count):
        raise ivy.utils.exceptions.IvyError(
            f"given feature_group_count {feature_group_count} expected input channel of"
            f" the filter to be {x.shape[-1] // feature_group_count} but got"
            f" {filters.shape[-2]}"
        )
    if x.shape[-1] % feature_group_count != 0:
        raise ivy.utils.exceptions.IvyError(
            "input channel should be divisible by feature group count"
            f" {feature_group_count} but got input channel {x.shape[-1]}"
        )

    if dims == 1:
        x, padding = _pad_before_conv(x, padding, dims)
        res = tf.nn.conv1d(
            x,
            filters,
            strides,
            padding,
            df,
            dilations,
        )
    elif dims == 2:
        padding, explicit_padding = _to_explicit_padding(padding, 2)
        strides = (
            [1] + list([strides] * 2 if isinstance(strides, int) else strides) + [1]
        )
        dilations = (
            [1]
            + list([dilations] * 2 if isinstance(dilations, int) else dilations)
            + [1]
        )
        res = tf.raw_ops.Conv2D(
            input=x,
            filter=filters,
            strides=strides,
            padding=padding,
            explicit_paddings=explicit_padding,
            data_format="NHWC",
            dilations=dilations,
        )
    else:
        x, padding = _pad_before_conv(x, padding, dims)
        strides = (
            [1] + list([strides] * 3 if isinstance(strides, int) else strides) + [1]
        )
        dilations = (
            [1]
            + list([dilations] * 3 if isinstance(dilations, int) else dilations)
            + [1]
        )
        # grouped conv3d is not supported on CPU
        # ToDO: change the condition of GPU when automatic device shifting
        #  is implemented in ivy
        if feature_group_count == 1 or tf.test.is_gpu_available():
            res = tf.nn.conv3d(
                x,
                filters,
                strides,
                padding,
                df,
                dilations,
            )
        else:
            res = tf.concat(
                [
                    tf.nn.conv3d(
                        x[:, :, :, :, i : i + filters.shape[-2]],
                        filters[
                            :, :, :, :, j : j + filters.shape[-1] // feature_group_count
                        ],
                        strides,
                        padding,
                        df,
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
    res = tf.math.add(res, bias) if bias is not None else res
    if data_format == "channel_first":
        res = tf.transpose(res, (0, dims + 1, *range(1, dims + 1)))
    return res


@with_unsupported_dtypes({"2.14.0 and below": ("bfloat16", "complex")}, backend_version)
def conv_general_transpose(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    bias: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "channel_first":
        x = tf.transpose(x, (0, *range(2, dims + 2), 1))
    if dims == 1:
        res = tf.concat(
            [
                conv1d_transpose(
                    x[..., j : j + filters.shape[-2] // feature_group_count],
                    filters[..., j : j + filters.shape[-2] // feature_group_count, :],
                    strides,
                    padding,
                    output_shape=output_shape,
                    data_format="NWC",
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
                conv2d_transpose(
                    x[..., j : j + filters.shape[-2] // feature_group_count],
                    filters[..., j : j + filters.shape[-2] // feature_group_count, :],
                    strides,
                    padding,
                    output_shape=output_shape,
                    data_format="NHWC",
                    dilations=dilations,
                )
                for j in range(
                    0, filters.shape[-2], filters.shape[-2] // feature_group_count
                )
            ],
            axis=-1,
        )
    else:
        res = tf.concat(
            [
                conv3d_transpose(
                    x[..., j : j + filters.shape[-2] // feature_group_count],
                    filters[..., j : j + filters.shape[-2] // feature_group_count, :],
                    strides,
                    padding,
                    output_shape=output_shape,
                    data_format="NDHWC",
                    dilations=dilations,
                )
                for j in range(
                    0, filters.shape[-2], filters.shape[-2] // feature_group_count
                )
            ],
            axis=-1,
        )
    res = tf.math.add(res, bias) if bias is not None else res
    if data_format == "channel_first":
        res = tf.transpose(res, (0, dims + 1, *range(1, dims + 1)))
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
