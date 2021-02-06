"""
Collection of MXNet network layers, wrapped to fit Ivy syntax and signature.
"""

# global
import math as _math
import mxnet as _mx


def conv1d(x, filters, strides, padding, data_format='NWC', dilations=1, filter_shape=None, num_filters=None):
    if data_format == 'NWC':
        x = _mx.nd.transpose(x, (0, 2, 1))
    if filter_shape is None:
        filter_shape = filters.shape[0:-2]
    if num_filters is None:
        num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == 'VALID':
        padding = [0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = _mx.nd.Convolution(data=x, weight=_mx.nd.transpose(filters, (1, 2, 0)), kernel=kernel, stride=strides, dilate=dilations, pad=padding, no_bias=True, num_filter=num_filters)
    if data_format == 'NWC':
        return _mx.nd.transpose(res, (0, 2, 1))
    else:
        return res


def conv1d_transpose(x, filters, strides, padding, _=None, data_format='NWC', dilations=1, filter_shape=None, num_filters=None):
    if data_format == 'NWC':
        x = _mx.nd.transpose(x, (0, 2, 1))
    if filter_shape is None:
        filter_shape = filters.shape[0:-2]
    if num_filters is None:
        num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == 'VALID':
        padding = [0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = _mx.nd.Deconvolution(data=x, weight=_mx.nd.transpose(filters, (1, 2, 0)), kernel=kernel, stride=strides, dilate=dilations, pad=padding, no_bias=True, num_filter=num_filters)
    if data_format == 'NWC':
        return _mx.nd.transpose(res, (0, 2, 1))
    else:
        return res


def conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1, filter_shape=None, num_filters=None):
    if data_format == 'NHWC':
        x = _mx.nd.transpose(x, (0, 3, 1, 2))
    if filter_shape is None:
        filter_shape = filters.shape[0:-2]
    if num_filters is None:
        num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == 'VALID':
        padding = [0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    strides = [strides]*2 if isinstance(strides, int) else strides
    dilations = [dilations]*2 if isinstance(dilations, int) else dilations
    res = _mx.nd.Convolution(data=x, weight=_mx.nd.transpose(filters, (2, 3, 0, 1)), kernel=kernel, stride=strides, dilate=dilations, pad=padding, no_bias=True, num_filter=num_filters)
    if data_format == 'NHWC':
        return _mx.nd.transpose(res, (0, 2, 3, 1))
    else:
        return res


def conv2d_transpose(x, filters, strides, padding, _=None, data_format='NHWC', dilations=1, filter_shape=None, num_filters=None):
    if data_format == 'NHWC':
        x = _mx.nd.transpose(x, (0, 3, 1, 2))
    if filter_shape is None:
        filter_shape = filters.shape[0:-2]
    if num_filters is None:
        num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == 'VALID':
        padding = [0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    strides = [strides]*2 if isinstance(strides, int) else strides
    dilations = [dilations]*2 if isinstance(dilations, int) else dilations
    res = _mx.nd.Deconvolution(data=x, weight=_mx.nd.transpose(filters, (2, 3, 0, 1)), kernel=kernel, stride=strides, dilate=dilations, pad=padding, no_bias=True, num_filter=num_filters)
    if data_format == 'NHWC':
        return _mx.nd.transpose(res, (0, 2, 3, 1))
    else:
        return res


def depthwise_conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1, filter_shape=None, num_filters=None, num_channels=None):
    if num_filters is None:
        num_filters = filters.shape[-1]
    if num_channels is None:
        num_channels = num_filters
    if data_format == 'NHWC':
        x = _mx.nd.transpose(x, (0, 3, 1, 2))
    if filter_shape is None:
        filter_shape = filters.shape[0:-1]
    kernel = filter_shape
    if padding == 'VALID':
        padding = [0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    strides = [strides]*2 if isinstance(strides, int) else strides
    dilations = [dilations]*2 if isinstance(dilations, int) else dilations
    res = _mx.nd.Convolution(data=x, weight=_mx.nd.transpose(_mx.nd.expand_dims(filters, -1), (2, 3, 0, 1)), kernel=kernel, stride=strides, dilate=dilations, pad=padding, no_bias=True, num_filter=num_filters, num_group=num_channels)
    if data_format == 'NHWC':
        return _mx.nd.transpose(res, (0, 2, 3, 1))
    else:
        return res


# noinspection PyDefaultArgument
def conv3d(x, filters, strides, padding, data_format='NDHWC', dilations=1, filter_shape=None, num_filters=None):
    if data_format == 'NDHWC':
        x = _mx.nd.transpose(x, (0, 4, 1, 2, 3))
    if filter_shape is None:
        filter_shape = filters.shape[0:-2]
    if num_filters is None:
        num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == 'VALID':
        padding = [0, 0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    strides = [strides]*3 if isinstance(strides, int) else strides
    dilations = [dilations]*3 if isinstance(dilations, int) else dilations
    res = _mx.nd.Convolution(data=x, weight=_mx.nd.transpose(filters, (3, 4, 0, 1, 2)), kernel=kernel, stride=strides, dilate=dilations, pad=padding, no_bias=True, num_filter=num_filters)
    if data_format == 'NDHWC':
        return _mx.nd.transpose(res, (0, 2, 3, 4, 1))
    else:
        return res


def conv3d_transpose(x, filters, strides, padding, _=None, data_format='NDHWC', dilations=1, filter_shape=None, num_filters=None):
    if data_format == 'NDHWC':
        x = _mx.nd.transpose(x, (0, 4, 1, 2, 3))
    if filter_shape is None:
        filter_shape = filters.shape[0:-2]
    if num_filters is None:
        num_filters = filters.shape[-1]
    kernel = filter_shape
    if padding == 'VALID':
        padding = [0, 0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    strides = [strides]*3 if isinstance(strides, int) else strides
    dilations = [dilations]*3 if isinstance(dilations, int) else dilations
    res = _mx.nd.Deconvolution(data=x, weight=_mx.nd.transpose(filters, (3, 4, 0, 1, 2)), kernel=kernel, stride=strides, dilate=dilations, pad=padding, no_bias=True, num_filter=num_filters)
    if data_format == 'NDHWC':
        return _mx.nd.transpose(res, (0, 2, 3, 4, 1))
    else:
        return res


def linear(x, weight, bias, num_hidden=None):
    num_hidden = weight.shape[0] if num_hidden is None else num_hidden
    return _mx.nd.FullyConnected(x, weight, bias, num_hidden=num_hidden)
