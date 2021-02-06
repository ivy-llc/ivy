"""
Collection of PyTorch network layers, wrapped to fit Ivy syntax and signature.
"""

# global
import math as _math
import torch as _torch


def conv1d(x, filters, strides, padding, data_format='NWC', dilations=1, filter_shape=None, _=None):
    if filter_shape is None:
        filter_shape = list(filters.shape[0:1])
    filters = filters.permute(1, 2, 0)
    if data_format == 'NWC':
        x = x.permute(0, 2, 1)
    if padding == 'VALID':
        padding = [0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = _torch.nn.functional.conv1d(x, filters, None, strides, padding, dilations)
    return res.permute(0, 2, 1)


def conv1d_transpose(x, filters, strides, padding, _=None, data_format='NWC', dilations=1, filter_shape=None, _1=None):
    if filter_shape is None:
        filter_shape = list(filters.shape[0:1])
    filters = filters.permute(1, 2, 0)
    if data_format == 'NWC':
        x = x.permute(0, 2, 1)
    if padding == 'VALID':
        padding = [0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = _torch.nn.functional.conv_transpose1d(x, filters, None, strides, padding, dilation=dilations)
    return res.permute(0, 2, 1)


def conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1, filter_shape=None, _=None):
    if filter_shape is None:
        filter_shape = list(filters.shape[0:2])
    filters = filters.permute(2, 3, 0, 1)
    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)
    if padding == 'VALID':
        padding = [0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = _torch.nn.functional.conv2d(x, filters, None, strides, padding, dilations)
    return res.permute(0, 2, 3, 1)


def conv2d_transpose(x, filters, strides, padding, _=None, data_format='NHWC', dilations=1, filter_shape=None, _1=None):
    if filter_shape is None:
        filter_shape = list(filters.shape[0:1])
    filters = filters.permute(2, 3, 0, 1)
    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)
    if padding == 'VALID':
        padding = [0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = _torch.nn.functional.conv_transpose2d(x, filters, None, strides, padding, dilation=dilations)
    return res.permute(0, 2, 3, 1)


def depthwise_conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1, filter_shape=None, _=None, _1=None):
    if filter_shape is None:
        filter_shape = list(filters.shape[0:2])
    dims_in = filters.shape[-1]
    filters = _torch.unsqueeze(filters, -1)
    filters = filters.permute(2, 3, 0, 1)
    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)
    if padding == 'VALID':
        padding = [0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    # noinspection PyArgumentEqualDefault
    res = _torch.nn.functional.conv2d(x, filters, None, strides, padding, dilations, dims_in)
    return res.permute(0, 2, 3, 1)


def conv3d(x, filters, strides, padding, data_format='NDHWC', dilations=1, filter_shape=None, _=None):
    if filter_shape is None:
        filter_shape = list(filters.shape[0:3])
    filters = filters.permute(3, 4, 0, 1, 2)
    if data_format == 'NDHWC':
        x = x.permute(0, 4, 1, 2, 3)
    if padding == 'VALID':
        padding = [0, 0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = _torch.nn.functional.conv3d(x, filters, None, strides, padding, dilations)
    return res.permute(0, 2, 3, 4, 1)


def conv3d_transpose(x, filters, strides, padding, _=None, data_format='NDHWC', dilations=1, filter_shape=None, _1=None):
    if filter_shape is None:
        filter_shape = list(filters.shape[0:1])
    filters = filters.permute(3, 4, 0, 1, 2)
    if data_format == 'NDHWC':
        x = x.permute(0, 4, 1, 2, 3)
    if padding == 'VALID':
        padding = [0, 0, 0]
    elif padding == 'SAME':
        padding = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = _torch.nn.functional.conv_transpose3d(x, filters, None, strides, padding, dilation=dilations)
    return res.permute(0, 2, 3, 4, 1)


linear = lambda x, weight, bias, _=None: _torch.nn.functional.linear(x, weight, bias)
