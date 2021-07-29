"""
Collection of Numpy linear algebra functions,
wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np

svd = _np.linalg.svd
norm = _np.linalg.norm
inv = _np.linalg.inv
pinv = _np.linalg.pinv


def vector_to_skew_symmetric_matrix(vector):
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = _np.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = _np.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = _np.concatenate((zs, -a3s, a2s), -1)
    row2 = _np.concatenate((a3s, zs, -a1s), -1)
    row3 = _np.concatenate((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return _np.concatenate((row1, row2, row3), -2)
