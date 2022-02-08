"""
Collection of Numpy reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np


def reduce_sum(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _np.asarray(_np.sum(x, axis=axis, keepdims=keepdims))
    if ret.shape == ():
        return ret.reshape((1,))
    return ret


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _np.asarray(_np.prod(x, axis=axis, keepdims=keepdims))
    if ret.shape == ():
        return ret.reshape((1,))
    return ret


def reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _np.asarray(_np.mean(x, axis=axis, keepdims=keepdims))
    if ret.shape == ():
        return ret.reshape((1,))
    return ret


def reduce_var(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _np.asarray(_np.var(x, axis=axis, keepdims=keepdims))
    if ret.shape == ():
        return ret.reshape((1,))
    return ret


def reduce_min(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _np.asarray(_np.min(x, axis=axis, keepdims=keepdims))
    if ret.shape == ():
        return ret.reshape((1,))
    return ret


def reduce_max(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _np.asarray(_np.max(x, axis=axis, keepdims=keepdims))
    if ret.shape == ():
        return ret.reshape((1,))
    return ret


def einsum(equation, *operands):
    ret = _np.asarray(_np.einsum(equation, *operands))
    if ret.shape == ():
        return ret.reshape((1,))
    return ret
