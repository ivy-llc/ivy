# global
from collections import namedtuple
from typing import Union, Optional, Tuple, Literal, Sequence, NamedTuple

import jax.numpy as jnp
import jax.scipy as jsp

# local
import ivy
from ivy import inf
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.jax import JaxArray

from . import backend_version

from ivy import promote_types_of_inputs


# Array API Standard #
# -------------------#


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def cholesky(
    x: JaxArray, /, *, upper: bool = False, out: Optional[JaxArray] = None
) -> JaxArray:
    if not upper:
        ret = jnp.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = jnp.transpose(jnp.linalg.cholesky(jnp.transpose(x, axes=axes)), axes=axes)
    return ret


@with_unsupported_dtypes({"0.4.13 and below": ("complex",)}, backend_version)
def cross(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.cross(a=x1, b=x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def det(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.linalg.det(x)


@with_unsupported_dtypes({"0.4.13 and below": ("float16", "bfloat16")}, backend_version)
def eig(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> Tuple[JaxArray]:
    result_tuple = NamedTuple(
        "eig", [("eigenvalues", JaxArray), ("eigenvectors", JaxArray)]
    )
    eigenvalues, eigenvectors = jnp.linalg.eig(x)
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_dtypes({"0.4.13 and below": ("complex",)}, backend_version)
def diagonal(
    x: JaxArray,
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if not x.dtype == bool and not jnp.issubdtype(x.dtype, jnp.integer):
        ret = jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)
        ret_edited = jnp.diagonal(
            x.at[1 / x == -jnp.inf].set(-jnp.inf),
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        )
        ret_edited = ret_edited.at[ret_edited == -jnp.inf].set(-0.0)
        ret = ret.at[ret == ret_edited].set(ret_edited[ret == ret_edited])
    else:
        ret = jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)
    return ret


def tensorsolve(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    axes: Optional[Union[int, Tuple[Sequence[int], Sequence[int]]]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.linalg.tensorsolve(x1, x2, axes)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def eigh(
    x: JaxArray, /, *, UPLO: str = "L", out: Optional[JaxArray] = None
) -> Tuple[JaxArray]:
    result_tuple = NamedTuple(
        "eigh", [("eigenvalues", JaxArray), ("eigenvectors", JaxArray)]
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(x, UPLO=UPLO)
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def eigvalsh(
    x: JaxArray, /, *, UPLO: str = "L", out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.linalg.eigvalsh(x, UPLO=UPLO)


@with_unsupported_dtypes({"0.4.13 and below": ("complex",)}, backend_version)
def inner(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.inner(x1, x2)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def inv(
    x: JaxArray,
    /,
    *,
    adjoint: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if adjoint:
        if x.ndim < 2:
            raise ValueError("Input must be at least 2D")
        permutation = list(range(x.ndim))
        permutation[-2], permutation[-1] = permutation[-1], permutation[-2]
        x_adj = jnp.transpose(x, permutation).conj()
        return jnp.linalg.inv(x_adj)
    return jnp.linalg.inv(x)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def matmul(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if transpose_a:
        x1 = jnp.swapaxes(x1, -1, -2)
    if transpose_b:
        x2 = jnp.swapaxes(x2, -1, -2)
    if adjoint_a:
        x1 = jnp.swapaxes(jnp.conjugate(x1), -1, -2)
    if adjoint_b:
        x2 = jnp.swapaxes(jnp.conjugate(x2), -1, -2)
    return jnp.matmul(x1, x2)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def matrix_norm(
    x: JaxArray,
    /,
    *,
    ord: Union[int, float, Literal[inf, -inf, "fro", "nuc"]] = "fro",
    axis: Tuple[int, int] = (-2, -1),
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if hasattr(axis, "__iter__"):
        if not isinstance(axis, tuple):
            axis = tuple(axis)
    else:
        if not isinstance(axis, tuple):
            axis = (axis,)
    return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


@with_unsupported_dtypes({"0.4.13 and below": ("complex",)}, backend_version)
def matrix_power(x: JaxArray, n: int, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.linalg.matrix_power(x, n)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def matrix_rank(
    x: JaxArray,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    hermitian: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if (x.ndim < 2) or (0 in x.shape):
        return jnp.asarray(0, jnp.int64)
    # we don't use the native matrix_rank function because the behaviour of the
    # tolerance argument is difficult to unify,
    # and the native implementation is compositional
    svd_values = jnp.linalg.svd(x, hermitian=hermitian, compute_uv=False)
    sigma = jnp.max(svd_values, axis=-1, keepdims=False)
    atol = (
        atol if atol is not None else jnp.finfo(x.dtype).eps * max(x.shape[-2:]) * sigma
    )
    rtol = rtol if rtol is not None else 0.0
    tol = jnp.maximum(atol, rtol * sigma)
    # make sure it's broadcastable again with svd_values
    tol = jnp.expand_dims(tol, axis=-1)
    ret = jnp.count_nonzero(svd_values > tol, axis=-1)
    return ret


@with_unsupported_dtypes(
    {"0.4.13 and below": ("int", "float16", "complex")},
    backend_version,
)
def matrix_transpose(
    x: JaxArray, /, *, conjugate: bool = False, out: Optional[JaxArray] = None
) -> JaxArray:
    if conjugate:
        x = jnp.conj(x)
    return jnp.swapaxes(x, -1, -2)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def outer(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.outer(x1, x2)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def pinv(
    x: JaxArray,
    /,
    *,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if rtol is None:
        ret = jnp.linalg.pinv(x)
    else:
        ret = jnp.linalg.pinv(x, rtol)
    return ret


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def qr(
    x: JaxArray, /, *, mode: str = "reduced", out: Optional[JaxArray] = None
) -> Tuple[JaxArray, JaxArray]:
    res = namedtuple("qr", ["Q", "R"])
    q, r = jnp.linalg.qr(x, mode=mode)
    return res(q, r)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def slogdet(
    x: JaxArray,
    /,
) -> Tuple[JaxArray, JaxArray]:
    results = NamedTuple("slogdet", [("sign", JaxArray), ("logabsdet", JaxArray)])
    sign, logabsdet = jnp.linalg.slogdet(x)
    return results(sign, logabsdet)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def solve(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    adjoint: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if adjoint:
        x1 = jnp.transpose(jnp.conjugate(x1))
    expanded_last = False
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if len(x2.shape) <= 1:
        if x2.shape[-1] == x1.shape[-1]:
            expanded_last = True
            x2 = jnp.expand_dims(x2, axis=1)

    # if any of the arrays are empty
    is_empty_x1 = x1.size == 0
    is_empty_x2 = x2.size == 0
    if is_empty_x1 or is_empty_x2:
        for i in range(len(x1.shape) - 2):
            x2 = jnp.expand_dims(x2, axis=0)
        output_shape = list(jnp.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
        output_shape.append(x2.shape[-2])
        output_shape.append(x2.shape[-1])
        ret = jnp.array([]).reshape(output_shape)
    else:
        output_shape = tuple(jnp.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
        x1 = jnp.broadcast_to(x1, output_shape + x1.shape[-2:])
        x2 = jnp.broadcast_to(x2, output_shape + x2.shape[-2:])
        ret = jnp.linalg.solve(x1, x2)

    if expanded_last:
        ret = jnp.squeeze(ret, axis=-1)
    return jnp.asarray(ret, dtype=x1.dtype)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def svd(
    x: JaxArray, /, *, compute_uv: bool = True, full_matrices: bool = True
) -> Union[JaxArray, Tuple[JaxArray, ...]]:
    if compute_uv:
        results = namedtuple("svd", "U S Vh")
        U, D, VT = jnp.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
        return results(U, D, VT)
    else:
        results = namedtuple("svd", "S")
        D = jnp.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
        return results(D)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def svdvals(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.linalg.svd(x, compute_uv=False)


@with_unsupported_dtypes({"0.4.13 and below": ("complex",)}, backend_version)
def tensordot(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.tensordot(x1, x2, axes)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def trace(
    x: JaxArray,
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.trace(x, offset=offset, axis1=axis1, axis2=axis2, out=out)


@with_unsupported_dtypes({"0.4.13 and below": ("complex",)}, backend_version)
def vecdot(
    x1: JaxArray, x2: JaxArray, /, *, axis: int = -1, out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.tensordot(x1, x2, axes=(axis, axis))


@with_unsupported_dtypes({"0.4.13 and below": ("complex",)}, backend_version)
def vector_norm(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    out: Optional[JaxArray] = None,
    dtype: Optional[jnp.dtype] = None,
) -> JaxArray:
    if dtype and x.dtype != dtype:
        x = x.astype(dtype)
    abs_x = jnp.abs(x)
    if ord == 0:
        return jnp.sum(
            (abs_x != 0).astype(abs_x.dtype), axis=axis, keepdims=keepdims, out=out
        )
    elif ord == inf:
        return jnp.max(abs_x, axis=axis, keepdims=keepdims, out=out)
    elif ord == -inf:
        return jnp.min(abs_x, axis=axis, keepdims=keepdims, out=out)
    else:
        return jnp.sum(abs_x**ord, axis=axis, keepdims=keepdims) ** (1.0 / ord)


# Extra #
# ------#


@with_unsupported_dtypes({"0.4.13 and below": ("complex",)}, backend_version)
def diag(
    x: JaxArray,
    /,
    *,
    k: int = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.diag(x, k=k)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def vander(
    x: JaxArray,
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.vander(x, N=N, increasing=increasing)


@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
            "complex",
            "unsigned",
        )
    },
    backend_version,
)
def vector_to_skew_symmetric_matrix(
    vector: JaxArray, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = jnp.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = jnp.zeros(batch_shape + [1, 1], dtype=vector.dtype)
    # BS x 1 x 3
    row1 = jnp.concatenate((zs, -a3s, a2s), -1)
    row2 = jnp.concatenate((a3s, zs, -a1s), -1)
    row3 = jnp.concatenate((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return jnp.concatenate((row1, row2, row3), -2)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def lu(
    A: JaxArray,
    /,
    *,
    pivot: bool = True,
    permute_l: bool = False,
    out: Optional[Union[Tuple[JaxArray, JaxArray], Tuple[JaxArray, JaxArray, JaxArray]]] = None
) -> Union[Tuple[JaxArray, JaxArray], Tuple[JaxArray, JaxArray, JaxArray]]:
    dtype = A.dtype
    res = namedtuple("PLU", ["P", "L", "U"])
    res2 = namedtuple("PLU", ["PL", "U"])
    P, L, U = jsp.linalg.lu(A)
    P = P.astype(dtype)
    L = L.astype(dtype)
    U = U.astype(dtype)
    if permute_l:
        return res2(jnp.matmul(P, L), U)
    return res(P, L, U)
