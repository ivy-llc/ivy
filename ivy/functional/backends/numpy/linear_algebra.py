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


@with_unsupported_dtypes({"1.23.0 and below": ("float16", "complex")}, backend_version)
def cholesky(
    x: np.ndarray, /, *, upper: bool = False, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not upper:
        ret = np.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = np.transpose(np.linalg.cholesky(np.transpose(x, axes=axes)), axes=axes)
    return ret


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def cross(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.cross(a=x1, b=x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
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


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def eigh(
    x: np.ndarray, /, *, UPLO: Optional[str] = "L", out: Optional[np.ndarray] = None
) -> Tuple[np.ndarray]:
    result_tuple = NamedTuple(
        "eigh", [("eigenvalues", np.ndarray), ("eigenvectors", np.ndarray)]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(x, UPLO=UPLO)
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def eigvalsh(
    x: np.ndarray, /, *, UPLO: Optional[str] = "L", out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.linalg.eigvalsh(x, UPLO=UPLO)


@_scalar_output_to_0d_array
def inner(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.inner(x1, x2)


@with_unsupported_dtypes(
    {
        "1.23.0 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def inv(
    x: np.ndarray,
    /,
    *,
    adjoint: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if np.any(np.linalg.det(x.astype("float64")) == 0):
        return x
    else:
        if adjoint is False:
            ret = np.linalg.inv(x)
            return ret
        else:
            x = np.transpose(x)
            ret = np.linalg.inv(x)
            return ret


@with_unsupported_dtypes({"1.23.0 and below": ("float16", "bfloat16")}, backend_version)
def matmul(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if transpose_a is True:
        x1 = np.transpose(x1)
    if transpose_b is True:
        x2 = np.transpose(x2)
    ret = np.matmul(x1, x2, out=out)
    if len(x1.shape) == len(x2.shape) == 1:
        ret = np.array(ret)
    return ret


matmul.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.23.0 and below": ("float16", "bfloat16")}, backend_version)
def matrix_norm(
    x: np.ndarray,
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    axis: Optional[Tuple[int, int]] = (-2, -1),
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
    {
        "1.23.0 and below": (
            "float16",
            "bfloat16",
            "complex"
        )
    },
    backend_version,
)
@_scalar_output_to_0d_array
def matrix_rank(
    x: np.ndarray,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    def dim_reduction(array):
        if array.ndim == 1:
            ret = array[0]
        elif array.ndim == 2:
            ret = array[0][0]
        elif array.ndim == 3:
            ret = array[0][0][0]
        elif array.ndim == 4:
            ret = array[0][0][0][0]
        return ret

    if len(x.shape) == 3:
        if x.shape[-3] == 0:
            return np.asarray(0).astype(x.dtype)
    elif len(x.shape) > 3:
        if x.shape[-3] == 0 or x.shape[-4] == 0:
            return np.asarray(0).astype(x.dtype)
    axis = None
    ret_shape = x.shape[:-2]
    if len(x.shape) == 2:
        singular_values = np.linalg.svd(x, compute_uv=False)
    elif len(x.shape) > 2:
        y = x.reshape((-1, *x.shape[-2:]))
        singular_values = np.asarray(
            [
                np.linalg.svd(split[0], compute_uv=False)
                for split in np.split(y, y.shape[0], axis=0)
            ]
        )
        axis = 1
    if len(x.shape) < 2 or len(singular_values.shape) == 0:
        return np.array(0, dtype=x.dtype)
    max_values = np.max(singular_values, axis=axis)
    if atol is None:
        if rtol is None:
            ret = np.sum(singular_values != 0, axis=axis)
        else:
            try:
                max_rtol = max_values * rtol
            except ValueError:
                if ivy.all(
                    element == rtol[0] for element in rtol
                ):  # all elements are same in rtol
                    rtol = dim_reduction(rtol)
                    max_rtol = max_values * rtol
            if not isinstance(rtol, float) and rtol.size > 1:
                if ivy.all(element == max_rtol[0] for element in max_rtol):
                    max_rtol = dim_reduction(max_rtol)
            elif not isinstance(max_values, float) and max_values.size > 1:
                if ivy.all(element == max_values[0] for element in max_values):
                    max_rtol = dim_reduction(max_rtol)
            ret = ivy.sum(singular_values > max_rtol, axis=axis)
    else:  # atol is not None
        if rtol is None:  # atol is not None, rtol is None
            ret = np.sum(singular_values > atol, axis=axis)
        else:
            tol = np.max(atol, max_values * rtol)
            ret = np.sum(singular_values > tol, axis=axis)
    if len(ret_shape):
        ret = ret.reshape(ret_shape)
    return ret.astype(x.dtype)


def matrix_transpose(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.swapaxes(x, -1, -2)


def outer(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.outer(x1, x2, out=out)


outer.support_native_out = True


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
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


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
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


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
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


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def solve(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
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


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
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


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def svdvals(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.linalg.svd(x, compute_uv=False)


def tensorsolve(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    axes: Union[int, Tuple[List[int], List[int]]] = None,
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
@with_unsupported_dtypes({"1.23.0 and below": ("float16", "bfloat16")}, backend_version)
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


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
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
    keepdims: Optional[bool] = False,
    ord: Optional[Union[int, float, Literal[inf, -inf]]] = 2,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dtype and x.dtype != dtype:
        x = x.astype(dtype)
    if isinstance(axis, list):
        axis = tuple(axis)
    if axis is None:
        np_normalized_vector = np.linalg.norm(x.flatten(), ord, axis, keepdims)
    else:
        if isinstance(ord, (int, float)) and ord != 0:
            np_normalized_vector = np.sum(
                np.abs(x) ** ord, axis=axis, keepdims=keepdims
            ) ** (1.0 / ord)
        else:
            np_normalized_vector = np.linalg.norm(x, ord, axis, keepdims)
    if np_normalized_vector.shape == ():
        np_normalized_vector = np.expand_dims(np_normalized_vector, 0)
    np_normalized_vector = np_normalized_vector.astype(x.dtype)
    return np_normalized_vector


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


@with_unsupported_dtypes({"1.23.0 and below": ("complex")}, backend_version)
def vander(
    x: np.ndarray,
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.vander(x, N=N, increasing=increasing).astype(x.dtype)


@with_unsupported_dtypes({"1.23.0 and below": ("complex")}, backend_version)
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
