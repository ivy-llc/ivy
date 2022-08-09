"""Collection of TensorFlow network layers, wrapped to fit Ivy syntax and signature."""

# global
import tensorflow as tf
from typing import Optional, Tuple, Union, List, Sequence
from tensorflow.python.types.core import Tensor

# local
import ivy


def _deconv_length(dim_size, stride_size, kernel_size, padding, dilation=1):

    # Get the dilated kernel size
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

    if padding == "VALID":
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    elif padding == "SAME":
        dim_size = dim_size * stride_size

    return dim_size


def conv1d(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: int,
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    dilations: int = 1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "NCW":
        x = tf.transpose(x, (0, 2, 1))
    res = tf.nn.conv1d(x, filters, strides, padding, "NWC", dilations)
    if data_format == "NCW":
        res = tf.transpose(res, (0, 2, 1))
    return res


def conv1d_transpose(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: int,
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NWC",
    dilations: int = 1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    if not ivy.gpu_is_available() and dilations > 1:
        raise Exception(
            "Tensorflow does not support dilations greater than 1 when device is cpu"
        )
    if data_format == "NCW":
        x = tf.transpose(x, (0, 2, 1))
    if output_shape is None:
        output_shape = list(
            _deconv_length(x.shape[1], strides, filters.shape[0], padding, dilations)
        )
    res = tf.nn.conv1d_transpose(
        x,
        filters,
        [x.shape[0], output_shape[0], x.shape[-1]],
        strides,
        padding,
        "NWC",
        dilations,
    )
    if data_format == "NCW":
        res = tf.transpose(res, (0, 2, 1))
    return res


def conv2d(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    dilations: int = 1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    res = tf.nn.conv2d(x, filters, strides, padding, "NHWC", dilations)
    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


def conv2d_transpose(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NHWC",
    dilations=1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    if isinstance(strides, int):
        strides = [strides] * 2
    elif len(strides) == 1:
        strides = (strides[0]) * 2
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    if not ivy.gpu_is_available() and (dilations[0] > 1 or dilations[1] > 1):
        raise Exception(
            "conv2d_transpose does not support dilations greater than 1 when device"
            "is cpu for tensorflow"
        )
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    if output_shape is None:
        new_h = _deconv_length(
            x.shape[1], strides[0], filters.shape[0], padding, dilations[0]
        )
        new_w = _deconv_length(
            x.shape[2], strides[1], filters.shape[1], padding, dilations[1]
        )
        output_shape = [new_h, new_w]
    output_shape = [x.shape[0]] + output_shape + [x.shape[-1]]
    res = tf.nn.conv2d_transpose(
        x, filters, output_shape, strides, padding, "NHWC", dilations
    )
    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


def depthwise_conv2d(
    x: Union[tf.Tensor, tf.Variable],
    filters: Union[tf.Tensor, tf.Variable],
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, List[int]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    if (
        not ivy.gpu_is_available()
        and (dilations[0] > 1 or dilations[1] > 1)
        and (strides[0] > 1 or strides[1] > 1)
    ):
        raise Exception(
            "depthwise_conv2d does not support dilations greater than 1 and"
            "strides greater than 1 when device is cpu for tensorflow"
        )
    filters = tf.expand_dims(filters, -1)
    strides = [1, strides[0], strides[1], 1]
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    res = tf.nn.depthwise_conv2d(x, filters, strides, padding, "NHWC", dilations)
    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


# noinspection PyDefaultArgument
def conv3d(
    x,
    filters,
    strides,
    padding,
    /,
    *,
    data_format="NDHWC",
    dilations=1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    strides = [1] + ([strides] * 3 if isinstance(strides, int) else strides) + [1]
    dilations = (
        [1] + ([dilations] * 3 if isinstance(dilations, int) else dilations) + [1]
    )
    if data_format == "NCDHW":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    res = tf.nn.conv3d(x, filters, strides, padding, "NDHWC", dilations)
    if data_format == "NCDHW":
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res


conv3d.unsupported_devices = ("cpu",)


def conv3d_transpose(
    x: Tensor,
    filters: Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NDHWC",
    dilations: int = 1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tensor:
    if not ivy.gpu_is_available() and (
        dilations[0] > 1 or dilations[1] > 1 or dilations[2] > 1
    ):
        raise Exception(
            "conv3d_transpose does not support dilations greater than 1 when"
            "device is cpu for tensorflow"
        )
    strides = [1] + ([strides] * 3 if isinstance(strides, int) else strides) + [1]
    dilations = (
        [1] + ([dilations] * 3 if isinstance(dilations, int) else dilations) + [1]
    )
    if data_format == "NCDHW":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    if output_shape is None:
        new_d = _deconv_length(
            x.shape[1], strides[0], filters.shape[0], padding, dilations[0]
        )
        new_h = _deconv_length(
            x.shape[2], strides[1], filters.shape[1], padding, dilations[1]
        )
        new_w = _deconv_length(
            x.shape[3], strides[2], filters.shape[2], padding, dilations[2]
        )
        output_shape = [new_d, new_h, new_w]
    output_shape = [x.shape[0]] + output_shape + [x.shape[-1]]
    res = tf.nn.conv3d_transpose(
        x, filters, output_shape, strides, padding, "NDHWC", dilations
    )
    if data_format == "NCDHW":
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res
