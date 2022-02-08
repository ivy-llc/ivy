"""
Collection of Jax linear algebra functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax.numpy as _jnp

svd = _jnp.linalg.svd


def matrix_norm(x, p=2, axes=None, keepdims=False):
    axes = (-2, -1) if axes is None else axes
    if isinstance(axes, int):
        raise Exception('if specified, axes must be a length-2 sequence of ints,'
                        'but found {} of type {}'.format(axes, type(axes)))
    elif isinstance(axes, list):
        axes = tuple(axes)
    ret = _jnp.linalg.norm(x, p, axes, keepdims)
    if ret.shape == ():
        return _jnp.expand_dims(ret, 0)
    return ret


inv = _jnp.linalg.inv
pinv = _jnp.linalg.pinv


def vector_to_skew_symmetric_matrix(vector):
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = _jnp.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = _jnp.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = _jnp.concatenate((zs, -a3s, a2s), -1)
    row2 = _jnp.concatenate((a3s, zs, -a1s), -1)
    row3 = _jnp.concatenate((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return _jnp.concatenate((row1, row2, row3), -2)
