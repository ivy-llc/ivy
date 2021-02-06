"""
Collection of Numpy network layers, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np


def conv1d(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv1d_transpose(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv2d(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def depthwise_conv2d(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv2d_transpose(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv3d(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


def conv3d_transpose(*_):
    raise Exception('Convolutions not yet implemented for numpy library')


linear = lambda x, weight, bias, _=None: _np.matmul(x, _np.transpose(weight)) + bias
