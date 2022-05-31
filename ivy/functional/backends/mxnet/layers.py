"""Collection of MXNet network layers, wrapped to fit Ivy syntax and signature."""

# global
import math
import mxnet as mx
from typing import Optional, Tuple, Union


def conv1d(
    x: mx.nd.NDArray,
    filters: mx.nd.NDArray,
    strides: int,
    padding: str,
    data_format: str = "NWC",
    dilations: int = 1,
) -> mx.nd.NDArray:
    if data_format == "NWC":
        x = mx.nd.transpose(x, (0, 2, 1))
    filter_shape = filters.shape[0:-2]
    num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == "VALID":
        padding = [0]
    elif padding == "SAME":
        padding = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = mx.nd.Convolution(
        data=x,
        weight=mx.nd.transpose(filters, (1, 2, 0)),
        kernel=kernel,
        stride=strides,
        dilate=dilations,
        pad=padding,
        no_bias=True,
        num_filter=num_filters,
    )
    if data_format == "NWC":
        return mx.nd.transpose(res, (0, 2, 1))
    else:
        return res


def conv1d_transpose(
    x, filters, strides, padding, _=None, data_format="NWC", dilations=1
):
    if data_format == "NWC":
        x = mx.nd.transpose(x, (0, 2, 1))
    filter_shape = filters.shape[0:-2]
    num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == "VALID":
        padding = [0]
    elif padding == "SAME":
        padding = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = mx.nd.Deconvolution(
        data=x,
        weight=mx.nd.transpose(filters, (1, 2, 0)),
        kernel=kernel,
        stride=strides,
        dilate=dilations,
        pad=padding,
        no_bias=True,
        num_filter=num_filters,
    )
    if data_format == "NWC":
        return mx.nd.transpose(res, (0, 2, 1))
    else:
        return res


def conv2d(x, filters, strides, padding, data_format="NHWC", dilations=1):
    if data_format == "NHWC":
        x = mx.nd.transpose(x, (0, 3, 1, 2))
    filter_shape = filters.shape[0:-2]
    num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == "VALID":
        padding = [0, 0]
    elif padding == "SAME":
        padding = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    res = mx.nd.Convolution(
        data=x,
        weight=mx.nd.transpose(filters, (2, 3, 0, 1)),
        kernel=kernel,
        stride=strides,
        dilate=dilations,
        pad=padding,
        no_bias=True,
        num_filter=num_filters,
    )
    if data_format == "NHWC":
        return mx.nd.transpose(res, (0, 2, 3, 1))
    else:
        return res


def conv2d_transpose(
    x, filters, strides, padding, _=None, data_format="NHWC", dilations=1
):
    if data_format == "NHWC":
        x = mx.nd.transpose(x, (0, 3, 1, 2))
    filter_shape = filters.shape[0:-2]
    num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == "VALID":
        padding = [0, 0]
    elif padding == "SAME":
        padding = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    res = mx.nd.Deconvolution(
        data=x,
        weight=mx.nd.transpose(filters, (2, 3, 0, 1)),
        kernel=kernel,
        stride=strides,
        dilate=dilations,
        pad=padding,
        no_bias=True,
        num_filter=num_filters,
    )
    if data_format == "NHWC":
        return mx.nd.transpose(res, (0, 2, 3, 1))
    else:
        return res


def depthwise_conv2d(
    x: mx.nd.NDArray,
    filters: mx.nd.NDArray,
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1
) -> mx.nd.NDArray:
    num_filters = filters.shape[-1]
    num_channels = num_filters
    if data_format == "NHWC":
        x = mx.nd.transpose(x, (0, 3, 1, 2))
    filter_shape = filters.shape[0:-1]
    kernel = filter_shape
    if padding == "VALID":
        padding = [0, 0]
    elif padding == "SAME":
        padding = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    res = mx.nd.Convolution(
        data=x,
        weight=mx.nd.transpose(mx.nd.expand_dims(filters, -1), (2, 3, 0, 1)),
        kernel=kernel,
        stride=strides,
        dilate=dilations,
        pad=padding,
        no_bias=True,
        num_filter=num_filters,
        num_group=num_channels,
    )
    if data_format == "NHWC":
        return mx.nd.transpose(res, (0, 2, 3, 1))
    else:
        return res


# noinspection PyDefaultArgument
def conv3d(x, filters, strides, padding, data_format="NDHWC", dilations=1):
    if data_format == "NDHWC":
        x = mx.nd.transpose(x, (0, 4, 1, 2, 3))
    filter_shape = filters.shape[0:-2]
    num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == "VALID":
        padding = [0, 0, 0]
    elif padding == "SAME":
        padding = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    res = mx.nd.Convolution(
        data=x,
        weight=mx.nd.transpose(filters, (3, 4, 0, 1, 2)),
        kernel=kernel,
        stride=strides,
        dilate=dilations,
        pad=padding,
        no_bias=True,
        num_filter=num_filters,
    )
    if data_format == "NDHWC":
        return mx.nd.transpose(res, (0, 2, 3, 4, 1))
    else:
        return res


def conv3d_transpose(
    x: mx.nd.NDArray,
    filters: mx.nd.NDArray,
    strides: Union[int, Tuple[int], Tuple[int,int], Tuple[int,int,int]],
    padding: str = None,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int], Tuple[int,int], Tuple[int,int,int]] = 1,
) -> mx.nd.NDArray:
    if data_format == "NDHWC":
        x = mx.nd.transpose(x, (0, 4, 1, 2, 3))
    filter_shape = filters.shape[0:-2]
    num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == "VALID":
        padding = [0, 0, 0]
    elif padding == "SAME":
        padding = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    res = mx.nd.Deconvolution(
        data=x,
        weight=mx.nd.transpose(filters, (3, 4, 0, 1, 2)),
        kernel=kernel,
        stride=strides,
        dilate=dilations,
        pad=padding,
        no_bias=True,
        num_filter=num_filters,
    )
    if data_format == "NDHWC":
        return mx.nd.transpose(res, (0, 2, 3, 4, 1))
    else:
        return res
