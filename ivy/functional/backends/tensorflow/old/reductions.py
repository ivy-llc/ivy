"""
Collection of TensorFlow reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _tf.reduce_prod(x, axis=axis, keepdims=keepdims)


def reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def reduce_var(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)


def einsum(equation, *operands):
    return _tf.einsum(equation, *operands)
