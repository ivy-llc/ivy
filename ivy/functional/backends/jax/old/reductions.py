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
    return _jnp.sum(x, axis=axis, keepdims=keepdims)


def einsum(equation, *operands):
    return _jnp.einsum(equation, *operands)

