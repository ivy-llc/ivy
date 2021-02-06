"""
Collection of MXNet reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx


def reduce_sum(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.nd.sum(x, axis=axis, keepdims=keepdims)


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.nd.prod(x, axis=axis, keepdims=keepdims)


def reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.nd.mean(x, axis=axis, keepdims=keepdims)


def reduce_min(x, axis=None, num_x_dims=None, keepdims=False):
    if num_x_dims is None:
        num_x_dims = len(x.shape)
    if axis is None:
        axis = list(range(num_x_dims))
    elif isinstance(axis, int):
        axis = [axis]
    axis = [(item + num_x_dims) % num_x_dims for item in axis]  # prevent negative indices
    axis.sort()
    return _mx.nd.min(x, axis=axis, keepdims=keepdims)


def reduce_max(x, axis=None, num_x_dims=None, keepdims=False):
    if num_x_dims is None:
        num_x_dims = len(x.shape)
    if axis is None:
        axis = list(range(num_x_dims))
    elif isinstance(axis, int):
        axis = [axis]
    axis = [(item + num_x_dims) % num_x_dims for item in axis]  # prevent negative indices
    axis.sort()
    return _mx.nd.max(x, axis=axis, keepdims=keepdims)
