# global
import numpy as np
from typing import Union, Optional, Tuple, Literal, List, NamedTuple

# local
import ivy
from ivy import inf
from collections import namedtuple


# Array API Standard #
# -------------------#
from ivy.functional.backends.numpy.helpers import _handle_0_dim_output


def cholesky(
    x: np.ndarray, /, *, upper: bool = False, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not upper:
        ret = np.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = np.transpose(np.linalg.cholesky(np.transpose(x, axes=axes)), axes=axes)
    return ret


cholesky.unsupported_dtypes = ("float16",)


def cross(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    axis: int = -1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.cross(a=x1, b=x2, axis=axis)


@_handle_0_dim_output
def det(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.det(x)


det.unsupported_dtypes = ("float16",)


def diagonal(
    x: np.ndarray,
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def eigh(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.eigh(x)


eigh.unsupported_dtypes = ("float16",)


def eigvalsh(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.eigvalsh(x)


eigvalsh.unsupported_dtypes = ("float16",)


def inv(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.inv(x)


inv.unsupported_dtypes = (
    "bfloat16",
    "float16",
)


def matmul(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    ret = np.matmul(x1, x2, out=out)
    if len(x1.shape) == len(x2.shape) == 1:
        ret = np.array(ret)
    return ret


matmul.support_native_out = True


@_handle_0_dim_output
def matrix_norm(
    x: np.ndarray,
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.linalg.norm(x, ord=ord, axis=(-2, -1), keepdims=keepdims)


matrix_norm.unsupported_dtypes = (
    "float16",
    "bfloat16",
)


def matrix_power(
    x: np.ndarray, n: int, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.linalg.matrix_power(x, n)


def matrix_rank(
    x: np.ndarray,
    /,
    *,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if rtol is None:
        ret = np.linalg.matrix_rank(x)
    else:
        ret = np.linalg.matrix_rank(x, rtol)
    return np.asarray(ret, dtype=ivy.default_int_dtype(as_native=True))


def matrix_transpose(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.swapaxes(x, -1, -2)


def outer(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.outer(x1, x2, out=out)


outer.support_native_out = True


def pinv(
    x: np.ndarray,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if rtol is None:
        return np.linalg.pinv(x)
    else:
        return np.linalg.pinv(x, rtol)


pinv.unsupported_dtypes = ("float16",)


def qr(x: np.ndarray, mode: str = "reduced") -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    q, r = np.linalg.qr(x, mode=mode)
    return res(q, r)


qr.unsupported_dtypes = ("float16",)


def slogdet(
    x: np.ndarray,
    *,
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = np.linalg.slogdet(x)
    sign = np.asarray(sign) if not isinstance(sign, np.ndarray) else sign
    logabsdet = (
        np.asarray(logabsdet) if not isinstance(logabsdet, np.ndarray) else logabsdet
    )
    return results(sign, logabsdet)


slogdet.unsupported_dtypes = ("float16",)


def solve(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    expanded_last = False
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if len(x2.shape) <= 1:
        if x2.shape[-1] == x1.shape[-1]:
            expanded_last = True
            x2 = np.expand_dims(x2, axis=1)
    for i in range(len(x1.shape) - 2):
        x2 = np.expand_dims(x2, axis=0)
    ret = np.linalg.solve(x1, x2)
    if expanded_last:
        ret = np.squeeze(ret, axis=-1)
    return ret


solve.unsupported_dtypes = ("float16",)


def svd(
    x: np.ndarray, full_matrices: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    results = namedtuple("svd", "U S Vh")
    U, D, VT = np.linalg.svd(x, full_matrices=full_matrices)
    return results(U, D, VT)


svd.unsupported_dtypes = ("float16",)


def svdvals(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.svd(x, compute_uv=False)


svdvals.unsupported_dtypes = ("float16",)


def tensordot(
    x1: np.ndarray,
    x2: np.ndarray,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.tensordot(x1, x2, axes=axes)


@_handle_0_dim_output
def trace(
    x: np.ndarray, offset: int = 0, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.trace(x, offset=offset, axis1=-2, axis2=-1, dtype=x.dtype, out=out)


trace.support_native_out = True


def vecdot(
    x1: np.ndarray, x2: np.ndarray, axis: int = -1, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.tensordot(x1, x2, axes=(axis, axis))


def vector_norm(
    x: np.ndarray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if axis is None:
        np_normalized_vector = np.linalg.norm(x.flatten(), ord, axis, keepdims)

    else:
        np_normalized_vector = np.linalg.norm(x, ord, axis, keepdims)

    if np_normalized_vector.shape == tuple():
        ret = np.expand_dims(np_normalized_vector, 0)
    else:
        ret = np_normalized_vector
    return ret


# Extra #
# ------#


def vector_to_skew_symmetric_matrix(
    vector: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = np.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = np.zeros(batch_shape + [1, 1], dtype=vector.dtype)
    # BS x 1 x 3
    row1 = np.concatenate((zs, -a3s, a2s), -1)
    row2 = np.concatenate((a3s, zs, -a1s), -1)
    row3 = np.concatenate((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return np.concatenate((row1, row2, row3), -2, out=out)


vector_to_skew_symmetric_matrix.support_native_out = True
