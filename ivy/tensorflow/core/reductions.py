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


def reduce_min(x, axis=None, num_x_dims=None, keepdims=False):
    if num_x_dims is None:
        num_x_dims = len(x.shape)
    if axis is None:
        axis = list(range(num_x_dims))
    elif isinstance(axis, int):
        axis = [axis]
    axis = [(item + num_x_dims) % num_x_dims for item in axis]  # prevent negative indices
    axis.sort()
    return _tf.reduce_min(x, axis=axis, keepdims=keepdims)


def reduce_max(x, axis=None, num_x_dims=None, keepdims=False):
    if num_x_dims is None:
        num_x_dims = len(x.shape)
    if axis is None:
        axis = list(range(num_x_dims))
    elif isinstance(axis, int):
        axis = [axis]
    axis = [(item + num_x_dims) % num_x_dims for item in axis]  # prevent negative indices
    axis.sort()
    return _tf.reduce_max(x, axis=axis, keepdims=keepdims)
