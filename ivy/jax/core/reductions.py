"""
Collection of Jax reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax.numpy as _jnp


def reduce_sum(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _jnp.sum(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _jnp.reshape(ret, (1,))
    return ret


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _jnp.prod(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _jnp.reshape(ret, (1,))
    return ret


def reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _jnp.mean(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _jnp.reshape(ret, (1,))
    return ret


def reduce_var(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _jnp.var(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _jnp.reshape(ret, (1,))
    return ret


def reduce_min(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _jnp.min(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _jnp.reshape(ret, (1,))
    return ret


def reduce_max(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    ret = _jnp.max(x, axis=axis, keepdims=keepdims)
    if ret.shape == ():
        return _jnp.reshape(ret, (1,))
    return ret


def einsum(equation, *operands):
    ret = _jnp.einsum(equation, *operands)
    if ret.shape == ():
        return _jnp.reshape(ret, (1,))
    return ret
