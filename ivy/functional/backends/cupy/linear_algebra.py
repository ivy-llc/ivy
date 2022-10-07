# global
import cupy as cp
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence

# local
import ivy
from ivy import inf
from collections import namedtuple


# Array API Standard #
# -------------------#
from ivy.functional.backends.numpy.helpers import _handle_0_dim_output


def cholesky(
    x: cp.ndarray, /, *, upper: Optional[bool] = False, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not upper:
        ret = cp.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = cp.transpose(cp.linalg.cholesky(cp.transpose(x, axes=axes)), axes=axes)
    return ret


cholesky.unsupported_dtypes = ("float16",)


def cross(
    x1: cp.ndarray,
    x2: cp.ndarray,
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.cross(a=x1, b=x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


@_handle_0_dim_output
def det(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.linalg.det(x)


det.unsupported_dtypes = ("float16",)


def diagonal(
    x: cp.ndarray,
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def eigh(
    x: cp.ndarray, /, *, UPLO: Optional[str] = "L", out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.linalg.eigh(x, UPLO=UPLO)


eigh.unsupported_dtypes = ("float16",)


def eigvalsh(
    x: cp.ndarray, /, *, UPLO: Optional[str] = "L", out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.linalg.eigvalsh(x)


eigvalsh.unsupported_dtypes = ("float16",)


@_handle_0_dim_output
def inner(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.inner(x1, x2)


def inv(
    x: cp.ndarray,
    /,
    *,
    adjoint: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if cp.any(cp.linalg.det(x.astype("float64")) == 0):
        return x
    else:
        if adjoint is False:
            ret = cp.linalg.inv(x)
            return ret
        else:
            x = cp.transpose(x)
            ret = cp.linalg.inv(x)
            return ret


inv.unsupported_dtypes = (
    "bfloat16",
    "float16",
)


def matmul(
    x1: cp.ndarray,
    x2: cp.ndarray,
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if transpose_a is True:
        x1 = cp.transpose(x1)
    if transpose_b is True:
        x2 = cp.transpose(x2)
    ret = cp.matmul(x1, x2, out=out)
    if len(x1.shape) == len(x2.shape) == 1:
        ret = cp.array(ret)
    return ret


matmul.support_native_out = True


@_handle_0_dim_output
def matrix_norm(
    x: cp.ndarray,
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.linalg.norm(x, ord=ord, axis=(-2, -1), keepdims=keepdims)


matrix_norm.unsupported_dtypes = (
    "float16",
    "bfloat16",
)


def matrix_power(
    x: cp.ndarray, n: int, /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.linalg.matrix_power(x, n)


@_handle_0_dim_output
def matrix_rank(
    x: cp.ndarray,
    /,
    *,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.asarray(cp.linalg.matrix_rank(x, tol=rtol)).astype(x.dtype)


matrix_rank.unsupported_dtypes = (
    "float16",
    "bfloat16",
)


def matrix_transpose(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.swapaxes(x, -1, -2)


def outer(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.outer(x1, x2, out=out)


outer.support_native_out = True


def pinv(
    x: cp.ndarray,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if rtol is None:
        return cp.linalg.pinv(x)
    else:
        return cp.linalg.pinv(x, rtol)


pinv.unsupported_dtypes = ("float16",)


def qr(x: cp.ndarray, mode: str = "reduced") -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    q, r = cp.linalg.qr(x, mode=mode)
    return res(q, r)


qr.unsupported_dtypes = ("float16",)


def slogdet(
    x: cp.ndarray,
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> Tuple[cp.ndarray, cp.ndarray]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = cp.linalg.slogdet(x)
    sign = cp.asarray(sign) if not isinstance(sign, cp.ndarray) else sign
    logabsdet = (
        cp.asarray(logabsdet) if not isinstance(logabsdet, cp.ndarray) else logabsdet
    )

    return results(sign, logabsdet)


slogdet.unsupported_dtypes = ("float16",)


def solve(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    expanded_last = False
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if len(x2.shape) <= 1:
        if x2.shape[-1] == x1.shape[-1]:
            expanded_last = True
            x2 = cp.expand_dims(x2, axis=1)
    for i in range(len(x1.shape) - 2):
        x2 = cp.expand_dims(x2, axis=0)
    ret = cp.linalg.solve(x1, x2)
    if expanded_last:
        ret = cp.squeeze(ret, axis=-1)
    return ret


solve.unsupported_dtypes = ("float16",)


def svd(
    x: cp.ndarray, full_matrices: bool = True
) -> Union[cp.ndarray, Tuple[cp.ndarray, ...]]:
    results = namedtuple("svd", "U S Vh")
    U, D, VT = cp.linalg.svd(x, full_matrices=full_matrices)
    return results(U, D, VT)


svd.unsupported_dtypes = ("float16",)


def svdvals(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.linalg.svd(x, compute_uv=False)


svdvals.unsupported_dtypes = ("float16",)


def tensordot(
    x1: cp.ndarray,
    x2: cp.ndarray,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.tensordot(x1, x2, axes=axes)


@_handle_0_dim_output
def trace(
    x: cp.ndarray, offset: int = 0, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.trace(x, offset=offset, axis1=-2, axis2=-1, dtype=x.dtype, out=out)


trace.support_native_out = True


def vecdot(
    x1: cp.ndarray, x2: cp.ndarray, axis: int = -1, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.tensordot(x1, x2, axes=(axis, axis))


def vector_norm(
    x: cp.ndarray,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if axis is None:
        np_normalized_vector = cp.linalg.norm(x.flatten(), ord, axis, keepdims)

    else:
        np_normalized_vector = cp.linalg.norm(x, ord, axis, keepdims)

    if np_normalized_vector.shape == tuple():
        ret = cp.expand_dims(np_normalized_vector, 0)
    else:
        ret = np_normalized_vector
    return ret


# Extra #
# ------#


def vector_to_skew_symmetric_matrix(
    vector: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = cp.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = cp.zeros(batch_shape + [1, 1], dtype=vector.dtype)
    # BS x 1 x 3
    row1 = cp.concatenate((zs, -a3s, a2s), -1)
    row2 = cp.concatenate((a3s, zs, -a1s), -1)
    row3 = cp.concatenate((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return cp.concatenate((row1, row2, row3), -2, out=out)


vector_to_skew_symmetric_matrix.support_native_out = True
