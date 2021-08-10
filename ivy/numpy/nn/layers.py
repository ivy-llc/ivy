"""
Collection of Numpy network layers, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np


def conv1d(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv1d_transpose(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1):
    filter_shape = filters.shape[0:2]
    filter_shape = list(filter_shape)
    if data_format == 'NCHW':
        x = _np.transpose(x, (0, 2, 3, 1))
    if padding == 'SAME':
        x = _np.pad(x, [[0, 0], [filter_shape[0]//2]*2, [filter_shape[1]//2]*2, [0, 0]])
    x_shape = x.shape
    input_dim = filters.shape[-2]
    output_dim = filters.shape[-1]
    new_h = x_shape[1] - filter_shape[0] + 1
    new_w = x_shape[2] - filter_shape[1] + 1
    new_shape = [x_shape[0], new_h, new_w] + filter_shape + [x_shape[-1]]
    # ToDo: add non-unit stride support
    new_strides = x.strides[0:1] + x.strides[1:3] + x.strides[1:3] + x.strides[-1:]
    # B x OH x OW x KH x KW x I
    sub_matrices = _np.lib.stride_tricks.as_strided(x, new_shape, new_strides, writeable=False)
    # B x OH x OW x KH x KW x I x O
    sub_matrices_w_output_dim = _np.tile(_np.expand_dims(sub_matrices, -1), [1]*6 + [output_dim])
    # B x OH x OW x KH x KW x I x O
    mult = sub_matrices_w_output_dim * filters.reshape([1]*3 + filter_shape + [input_dim, output_dim])
    # B x OH x OW x O
    res = _np.sum(mult, (3, 4, 5))
    if data_format == 'NCHW':
        return _np.transpose(res, (0, 3, 1, 2))
    return res


def depthwise_conv2d(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv2d_transpose(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv3d(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv3d_transpose(*_):
    raise Exception('Convolutions not yet implemented for numpy library')
