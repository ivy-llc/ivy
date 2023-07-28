# global

from collections import namedtuple

from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence


import numpy as np

# local
import ivy
from ivy import inf
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from . import backend_version


# Array API Standard #
# -------------------#


@with_unsupported_dtypes({"1.25.1 and below": ("float16", "complex")}, backend_version)
def cholesky(
    x: np.ndarray, /, *, upper: bool = False, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not upper:
        ret = np.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = np.transpose(np.linalg.cholesky(np.transpose(x, axes=axes)), axes=axes)
    return ret


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def cross(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.cross(a=x1, b=x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def det(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.det(x)


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


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def eigh(
    x: np.ndarray, /, *, UPLO: str = "L", out: Optional[np.ndarray] = None
) -> Tuple[np.ndarray]:
    result_tuple = NamedTuple(
        "eigh", [("eigenvalues", np.ndarray), ("eigenvectors", np.ndarray)]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(x, UPLO=UPLO)
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def eigvalsh(
    x: np.ndarray, /, *, UPLO: str = "L", out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.linalg.eigvalsh(x, UPLO=UPLO)


@_scalar_output_to_0d_array
def inner(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.inner(x1, x2)


@with_unsupported_dtypes(
    {"1.25.1 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def inv(
    x: np.ndarray,
    /,
    *,
    adjoint: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if adjoint:
        if x.ndim < 2:
            raise ValueError("Input must be at least 2D")
        permutation = list(range(x.ndim))
        permutation[-2], permutation[-1] = permutation[-1], permutation[-2]
        x_adj = np.transpose(x, permutation).conj()
        return np.linalg.inv(x_adj)
    return np.linalg.inv(x)


@with_unsupported_dtypes({"1.25.1 and below": ("float16", "bfloat16")}, backend_version)
def matmul(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if transpose_a:
        x1 = np.swapaxes(x1, -1, -2)
    if transpose_b:
        x2 = np.swapaxes(x2, -1, -2)
    if adjoint_a:
        x1 = np.swapaxes(np.conjugate(x1), -1, -2)
    if adjoint_b:
        x2 = np.swapaxes(np.conjugate(x2), -1, -2)
    ret = np.matmul(x1, x2, out=out)
    if len(x1.shape) == len(x2.shape) == 1:
        ret = np.array(ret)
    return ret


matmul.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.25.1 and below": ("float16", "bfloat16")}, backend_version)
def matrix_norm(
    x: np.ndarray,
    /,
    *,
    ord: Union[int, float, Literal[inf, -inf, "fro", "nuc"]] = "fro",
    axis: Tuple[int, int] = (-2, -1),
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not isinstance(axis, tuple):
        axis = tuple(axis)
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def matrix_power(
    x: np.ndarray, n: int, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.linalg.matrix_power(x, n)


@with_unsupported_dtypes(
    {"1.25.1 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
@_scalar_output_to_0d_array
def matrix_rank(
    x: np.ndarray,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    hermitian: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if (x.ndim < 2) or (0 in x.shape):
        return np.asarray(0, np.int64)
    # we don't use the native matrix_rank function because the behaviour of the
    # tolerance argument is difficult to unify,
    # and the native implementation is compositional
    svd_values = np.linalg.svd(x, hermitian=hermitian, compute_uv=False)
    sigma = np.max(svd_values, axis=-1, keepdims=False)
    atol = (
        atol if atol is not None else np.finfo(x.dtype).eps * max(x.shape[-2:]) * sigma
    )
    rtol = rtol if rtol is not None else 0.0
    tol = np.maximum(atol, rtol * sigma)
    # make sure it's broadcastable again with svd_values
    tol = np.expand_dims(tol, axis=-1)
    ret = np.count_nonzero(svd_values > tol, axis=-1)
    return ret


def matrix_transpose(
    x: np.ndarray, /, *, conjugate: bool = False, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if conjugate:
        x = np.conjugate(x)
    return np.swapaxes(x, -1, -2)


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def outer(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.outer(x1, x2, out=out)


outer.support_native_out = True


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def pinv(
    x: np.ndarray,
    /,
    *,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if rtol is None:
        return np.linalg.pinv(x)
    else:
        return np.linalg.pinv(x, rtol)


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def qr(
    x: np.ndarray,
    /,
    *,
    mode: str = "reduced",
    out: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    q, r = np.linalg.qr(x, mode=mode)
    return res(q, r)


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def slogdet(
    x: np.ndarray,
    /,
) -> Tuple[np.ndarray, np.ndarray]:
    results = NamedTuple("slogdet", [("sign", np.ndarray), ("logabsdet", np.ndarray)])
    sign, logabsdet = np.linalg.slogdet(x)
    sign = np.asarray(sign) if not isinstance(sign, np.ndarray) else sign
    logabsdet = (
        np.asarray(logabsdet) if not isinstance(logabsdet, np.ndarray) else logabsdet
    )

    return results(sign, logabsdet)


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def solve(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    adjoint: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if adjoint:
        x1 = np.transpose(np.conjugate(x1))
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


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def svd(
    x: np.ndarray, /, *, compute_uv: bool = True, full_matrices: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if compute_uv:
        results = namedtuple("svd", "U S Vh")
        U, D, VT = np.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
        return results(U, D, VT)
    else:
        results = namedtuple("svd", "S")
        D = np.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
        return results(D)


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def svdvals(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.svd(x, compute_uv=False)


def tensorsolve(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    axes: Optional[Union[int, Tuple[List[int], List[int]]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.linalg.tensorsolve(x1, x2, axes=axes)


def tensordot(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.tensordot(x1, x2, axes=axes)


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.25.1 and below": ("float16", "bfloat16")}, backend_version)
def trace(
    x: np.ndarray,
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.trace(x, offset=offset, axis1=axis1, axis2=axis2, out=out)


trace.support_native_out = True


def vecdot(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    axis: int = -1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.tensordot(x1, x2, axes=(axis, axis))


@with_unsupported_dtypes({"1.25.1 and below": ("float16",)}, backend_version)
def eig(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> Tuple[np.ndarray]:
    result_tuple = NamedTuple(
        "eig", [("eigenvalues", np.ndarray), ("eigenvectors", np.ndarray)]
    )
    eigenvalues, eigenvectors = np.linalg.eig(x)
    return result_tuple(eigenvalues, eigenvectors)


def vector_norm(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype and x.dtype != dtype:
        x = x.astype(dtype)
    abs_x = np.abs(x)
    if isinstance(axis, list):
        axis = tuple(axis)
    if ord == 0:
        return np.sum(
            (abs_x != 0).astype(abs_x.dtype), axis=axis, keepdims=keepdims, out=out
        )
    elif ord == inf:
        return np.max(abs_x, axis=axis, keepdims=keepdims, out=out)
    elif ord == -inf:
        return np.min(abs_x, axis=axis, keepdims=keepdims, out=out)
    else:
        return (
            np.sum(abs_x**ord, axis=axis, keepdims=keepdims) ** (1.0 / ord)
        ).astype(abs_x.dtype)


# Extra #
# ----- #


def diag(
    x: np.ndarray,
    /,
    *,
    k: int = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.diag(x, k=k)


@with_unsupported_dtypes({"1.24.0 and below": ("complex",)}, backend_version)
def vander(
    x: np.ndarray,
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.vander(x, N=N, increasing=increasing).astype(x.dtype)


@with_unsupported_dtypes(
    {
        "1.25.1 and below": (
            "complex",
            "unsigned",
        )
    },
    backend_version,
)
def vector_to_skew_symmetric_matrix(
    vector: np.ndarray, /, *, out: Optional[np.ndarray] = None
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

@with_unsupported_dtypes({"1.23.0 and below": ("uint64", "float16", "uint16", "bfloat16", "uint32")}, backend_version)
def lu(
    A: np.ndarray,
    /,
    *,
    pivot: bool = True,
    permute_l: bool = False,
    out: Optional[Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    dtype = A.dtype
    res = namedtuple("PLU", ["P", "L", "U"])
    res2 = namedtuple("PLU", ["PL", "U"])
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    for i in range(n):
        for k in range(i, n):
            if ~np.isclose(U[i, i], 0.0):
                break
            U[[k, k + 1]] = U[[k + 1, k]]
            P[[k, k + 1]] = P[[k + 1, k]]
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]
    P = P.astype(dtype)
    L = L.astype(dtype)
    U = U.astype(dtype)
    if permute_l:
        return res2(np.matmul(P, L), U)
    return res(P, L, U)
