"""
Collection of MXNet reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx


def reduce_sum(x, axis=None, keepdims=False):
    if axis is None:
        raise Exception('axis must be provided for calling ivy.reduce_sum() in mxnet symbolic mode.')
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.symbol.sum(x, axis=axis, keepdims=keepdims)


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        raise Exception('axis must be provided for calling ivy.reduce_prod() in mxnet symbolic mode.')
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.symbol.prod(x, axis=axis, keepdims=keepdims)


def reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        raise Exception('axis must be provided for calling ivy.reduce_mean() in mxnet symbolic mode.')
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.sym.mean(x, axis=axis, keepdims=keepdims)


def reduce_min(x, axis=None, num_x_dims=None, keepdims=False):
    if num_x_dims is None:
        raise Exception('num_x_dims must be provided for calling ivy.reduce_min() in mxnet symbolic mode.')
    if axis is None:
        axis = list(range(num_x_dims))
    elif isinstance(axis, int):
        axis = [axis]
    axis = [(item + num_x_dims) % num_x_dims for item in axis]  # prevent negative indices
    axis.sort()
    return _mx.symbol.min(x, axis=axis, keepdims=keepdims)


def reduce_max(x, axis=None, num_x_dims=None, keepdims=False):
    if num_x_dims is None:
        raise Exception('num_x_dims must be provided for calling ivy.reduce_max() in mxnet symbolic mode.')
    if axis is None:
        axis = list(range(num_x_dims))
    elif isinstance(axis, int):
        axis = [axis]
    axis = [(item + num_x_dims) % num_x_dims for item in axis]  # prevent negative indices
    axis.sort()
    return _mx.sym.max(x, axis=axis, keepdims=keepdims)
