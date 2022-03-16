"""
Collection of TensorFlow reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf


def reduce_sum(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _tf.reduce_sum(x, axis=axis, keepdims=keepdims)


def einsum(equation, *operands):
    return _tf.einsum(equation, *operands)
