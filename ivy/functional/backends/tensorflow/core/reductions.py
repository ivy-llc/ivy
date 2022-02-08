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
    ret = _tf.reduce_sum(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _tf.reshape(ret, (1,))
    return ret


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _tf.reduce_prod(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _tf.reshape(ret, (1,))
    return ret


def reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _tf.reduce_mean(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _tf.reshape(ret, (1,))
    return ret


def reduce_var(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _tf.reshape(ret, (1,))
    return ret


def reduce_min(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _tf.reduce_min(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _tf.reshape(ret, (1,))
    return ret


def reduce_max(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _tf.reduce_max(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _tf.reshape(ret, (1,))
    return ret


def einsum(equation, *operands):
    ret = _tf.einsum(equation, *operands)
    if ret.shape == ():
        return _tf.reshape(ret, (1,))
    return ret
