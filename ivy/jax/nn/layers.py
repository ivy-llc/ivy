"""
Collection of Jax network layers, wrapped to fit Ivy syntax and signature.
"""

# global
import jax.numpy as _jnp


def conv1d(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv1d_transpose(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv2d(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def depthwise_conv2d(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv2d_transpose(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv3d(*_):
    raise Exception('Convolutions not yet implemented for jax library')


def conv3d_transpose(*_):
    raise Exception('Convolutions not yet implemented for jax library')


linear = lambda x, weight, bias, _=None: _jnp.matmul(x, _jnp.transpose(weight)) + bias
