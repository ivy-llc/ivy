"""
Collection of Jax linear algebra functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax.numpy as _jnp

svd = lambda x, batch_shape=None: _jnp.linalg.svd(x)
norm = _jnp.linalg.norm
inv = _jnp.linalg.inv
pinv = _jnp.linalg.pinv


def vector_to_skew_symmetric_matrix(vector, batch_shape=None):
    if batch_shape is None:
        batch_shape = vector.shape[:-1]
    # shapes as list
    batch_shape = list(batch_shape)
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
