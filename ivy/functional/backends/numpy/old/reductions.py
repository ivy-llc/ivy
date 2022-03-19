"""
Collection of Numpy reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np
import numpy.array_api as _npa


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _np.asarray(_np.prod(x, axis=axis, keepdims=keepdims))


def reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _np.asarray(_np.mean(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _np.asarray(_np.var(x, axis=axis, keepdims=keepdims))


def einsum(equation, *operands):
    return _np.asarray(_np.einsum(equation, *operands))
