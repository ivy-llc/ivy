"""
Collection of TensorFlow linear algebra functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf


# noinspection PyPep8Naming
def svd(x):
    batch_shape = _tf.shape(x)[:-2]
    num_batch_dims = len(batch_shape)
    transpose_dims = list(range(num_batch_dims)) + [num_batch_dims + 1, num_batch_dims]
    D, U, V = _tf.linalg.svd(x)
    VT = _tf.transpose(V, transpose_dims)
    return U, D, VT


# noinspection PyShadowingBuiltins
norm = lambda x, ord=2, axis=-1, keepdims=False: _tf.linalg.norm(x, ord, axis, keepdims)
inv = _tf.linalg.inv
pinv = _tf.linalg.pinv


def vector_to_skew_symmetric_matrix(vector):
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = _tf.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = _tf.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = _tf.concat((zs, -a3s, a2s), -1)
    row2 = _tf.concat((a3s, zs, -a1s), -1)
    row3 = _tf.concat((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return _tf.concat((row1, row2, row3), -2)
