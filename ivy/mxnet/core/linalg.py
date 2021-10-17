"""
Collection of MXNet linear algebra functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

# local
import ivy as _ivy
from ivy.mxnet.core.general import matmul as _matmul


# noinspection PyPep8Naming
def svd(x):
    U, D, VT = _mx.nd.np.linalg.svd(x.as_np_ndarray())
    return U.as_nd_ndarray(), D.as_nd_ndarray(), VT.as_nd_ndarray()


def matrix_norm(x, p=2, axes=None, keepdims=False):
    axes = (-2, -1) if axes is None else axes
    if isinstance(axes, int):
        raise Exception('if specified, axes must be a length-2 sequence of ints,'
                        'but found {} of type {}'.format(axes, type(axes)))
    return _mx.nd.norm(x, p, axes, keepdims=keepdims)


inv = _mx.nd.linalg_inverse


DET_THRESHOLD = 1e-12


def pinv(x):
    """
    reference: https://help.matheass.eu/en/Pseudoinverse.html
    """
    x_dim, y_dim = x.shape[-2:]
    if x_dim == y_dim and _mx.nd.sum(_mx.nd.linalg.det(x) > DET_THRESHOLD) > 0:
        return inv(x)
    else:
        xT = _mx.nd.swapaxes(x, -1, -2)
        xT_x = _ivy.to_native(_matmul(xT, x))
        if _mx.nd.linalg.det(xT_x) > DET_THRESHOLD:
            return _matmul(inv(xT_x), xT)
        else:
            x_xT = _ivy.to_native(_matmul(x, xT))
            if _mx.nd.linalg.det(x_xT) > DET_THRESHOLD:
                return _matmul(xT, inv(x_xT))
            else:
                return xT


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
