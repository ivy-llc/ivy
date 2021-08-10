"""
Collection of Jax network layers, wrapped to fit Ivy syntax and signature.
"""

# global
import jax.numpy as _jnp
import jax.lax as _jlax


def conv1d(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv1d_transpose(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1):
    strides = [strides]*2 if isinstance(strides, int) else strides
    dilations = [dilations]*2 if isinstance(dilations, int) else dilations
    return _jlax.conv_general_dilated(x, filters, strides, padding, None, dilations, (data_format, 'HWIO', data_format))


def depthwise_conv2d(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv2d_transpose(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv3d(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv3d_transpose(*_):
    raise Exception('Convolutions not yet implemented for jax library')
