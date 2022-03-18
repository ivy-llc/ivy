"""
Collection of MXNet linear algebra functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx
import numpy as _np
# local
import ivy as _ivy
from typing import Union, Tuple





def matrix_norm(x, p=2, axes=None, keepdims=False):
    axes = (-2, -1) if axes is None else axes
    if isinstance(axes, int):
        raise Exception('if specified, axes must be a length-2 sequence of ints,'
                        'but found {} of type {}'.format(axes, type(axes)))
    return _mx.nd.norm(x, p, axes, keepdims=keepdims)


cholesky = lambda x: _mx.np.linalg.cholesky(x.as_np_ndarray()).as_nd_ndarray()



def vector_to_skew_symmetric_matrix(vector):
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = _mx.nd.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = _mx.nd.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = _mx.nd.concat(*(zs, -a3s, a2s), dim=-1)
    row2 = _mx.nd.concat(*(a3s, zs, -a1s), dim=-1)
    row3 = _mx.nd.concat(*(-a2s, a1s, zs), dim=-1)
    # BS x 3 x 3
    return _mx.nd.concat(*(row1, row2, row3), dim=-2)

def qr(x, mode):
    return _mx.np.linalg.qr(x, mode=mode)
