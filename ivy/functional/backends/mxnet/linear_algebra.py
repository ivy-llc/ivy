# global
import mxnet as mx
from collections import namedtuple
from mxnet.ndarray.ndarray import NDArray
from typing import Union, Optional, Tuple, Literal

# local
import ivy
from ivy import inf

DET_THRESHOLD = 1e-12


# Array API Standard #
# -------------------#


def cholesky(x: mx.nd.NDArray, upper: bool = False) -> mx.nd.NDArray:
    if not upper:
        return mx.np.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        return mx.np.transpose(
            mx.np.linalg.cholesky(mx.np.transpose(x, axes=axes)), axes=axes
        )


def cross(x1: mx.nd.NDArray, x2: mx.nd.NDArray, axis: int = -1) -> mx.nd.NDArray:
    return mx.np.cross(a=x1, b=x2, axis=axis)


def det(x: NDArray, out: Optional[NDArray] = None) -> NDArray:
    ret = mx.linalg.det(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


det.unsupported_dtypes = ("float16",)


def diagonal(x: NDArray, offset: int = 0, axis1: int = -2, axis2: int = -1) -> NDArray:
    return mx.nd.diag(x, k=offset, axis1=axis1, axis2=axis2)


def eigh(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.np.linalg.eigh(x)


def eigvalsh(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.np.linalg.eigvalsh(x)


def inv(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.linalg.inverse(x)


def matmul(x1, x2):
    expanded = False
    x1_shape = list(x1.shape)
    x2_shape = list(x2.shape)
    if len(x1_shape) != 3:
        num_x1_dims = len(x1_shape)
        x1 = mx.nd.reshape(
            x1, [1] * max(2 - num_x1_dims, 0) + [-1] + x1_shape[-min(num_x1_dims, 2) :]
        )
        expanded = True
    if len(x2_shape) != 3:
        num_x2_dims = len(x2_shape)
        x2 = mx.nd.reshape(
            x2, [1] * max(2 - num_x2_dims, 0) + [-1] + x2_shape[-min(num_x2_dims, 2) :]
        )
        expanded = True
    x1_batch_size = x1.shape[0]
    x2_batch_size = x2.shape[0]
    if x1_batch_size > x2_batch_size:
        x2 = mx.nd.tile(x2, (int(x1_batch_size / x2_batch_size), 1, 1))
    elif x2_batch_size > x1_batch_size:
        x1 = mx.nd.tile(x1, (int(x2_batch_size / x1_batch_size), 1, 1))
    res = mx.nd.batch_dot(x1, x2)
    if expanded:
        return mx.nd.reshape(res, list(x1_shape[:-1]) + [res.shape[-1]])
    return res


def matrix_norm(x, p=2, axes=None, keepdims=False):
    axes = (-2, -1) if axes is None else axes
    if isinstance(axes, int):
        raise Exception(
            "if specified, axes must be a length-2 sequence of ints,"
            "but found {} of type {}".format(axes, type(axes))
        )
    return mx.nd.norm(x, p, axes, keepdims=keepdims)


def matrix_rank(
    x: NDArray, rtol: Union[NDArray, float] = None
) -> Union[NDArray, float]:
    return mx.np.linalg.matrix_rank(x, rtol)


def matrix_transpose(x, axes=None):
    if axes is None:
        num_dims = len(x.shape)
        axes = list(range(num_dims))
        axes.reverse()
    return mx.nd.transpose(x, axes)


def outer(x1: mx.nd.NDArray, x2: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.outer(x1, x2)


def pinv(x):
    """reference: https://help.matheass.eu/en/Pseudoinverse.html"""
    x_dim, y_dim = x.shape[-2:]
    if x_dim == y_dim and mx.nd.sum(mx.nd.linalg.det(x) > DET_THRESHOLD) > 0:
        return inv(x)
    else:
        xT = mx.nd.swapaxes(x, -1, -2)
        xT_x = ivy.to_native(matmul(xT, x))
        if mx.nd.linalg.det(xT_x) > DET_THRESHOLD:
            return matmul(inv(xT_x), xT)
        else:
            x_xT = ivy.to_native(matmul(x, xT))
            if mx.nd.linalg.det(x_xT) > DET_THRESHOLD:
                return matmul(xT, inv(x_xT))
            else:
                return xT


def qr(x, mode):
    return mx.np.linalg.qr(x, mode=mode)


def slogdet(
    x: Union[ivy.Array, ivy.NativeArray], full_matrices: bool = True
) -> Union[ivy.Array, Tuple[ivy.Array, ...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = mx.linalg.slogdet(x)
    res = results(sign, logabsdet)

    return res


def svd(x: NDArray, full_matrices: bool = True) -> Union[NDArray, Tuple[NDArray, ...]]:
    return mx.np.linalg.svd(x)


def trace(x: NDArray, offset: int = 0) -> mx.np.ndarray:
    return mx.np.trace(x, offset=offset)


def vector_norm(
    x: NDArray,
    p: Union[int, float, Literal[inf, -inf]] = 2,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
) -> NDArray:
    return mx.np.linalg.norm(x, p, axis, keepdims)


# Extra #
# ------#


def vector_to_skew_symmetric_matrix(vector: NDArray) -> NDArray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = mx.nd.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = mx.nd.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = mx.nd.concat(*(zs, -a3s, a2s), dim=-1)
    row2 = mx.nd.concat(*(a3s, zs, -a1s), dim=-1)
    row3 = mx.nd.concat(*(-a2s, a1s, zs), dim=-1)
    # BS x 3 x 3
    return mx.nd.concat(*(row1, row2, row3), dim=-2)
