"""
Collection of MXNet reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx
from numbers import Number


def reduce_sum(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.nd.sum(x, axis=axis, keepdims=keepdims)


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.nd.prod(x, axis=axis, keepdims=keepdims)


def _reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.nd.mean(x, axis=axis, keepdims=keepdims)


def reduce_mean(x, axis=None, keepdims=False):
    return _reduce_mean(x, axis, keepdims)


def reduce_var(x, axis=None, keepdims=False):
    mean_of_x_sqrd = _reduce_mean(x ** 2, axis, keepdims)
    mean_of_x = _reduce_mean(x, axis, keepdims)
    return mean_of_x_sqrd - mean_of_x ** 2


def reduce_min(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.nd.min(x, axis=axis, keepdims=keepdims)


def reduce_max(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _mx.nd.max(x, axis=axis, keepdims=keepdims)


def einsum(equation, *operands):
    ret = _mx.np.einsum(equation, *[op.as_np_ndarray() for op in operands])
    if ret.shape == ():
        return _mx.np.resize(ret, (1,)).as_nd_ndarray()
    return ret.as_nd_ndarray()
