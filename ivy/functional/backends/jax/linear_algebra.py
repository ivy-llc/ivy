# global
from collections import namedtuple
from typing import Union, Optional, Tuple, Literal, Sequence, NamedTuple

import jax.numpy as jnp

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
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
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


@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
def cross(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.cross(a=x1, b=x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def det(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.linalg.det(x)

@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
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
        ret = jnp.diagonal(x, offset, axis1, axis2)
        ret_edited = jnp.diagonal(
            x.at[1 / x == -jnp.inf].set(-jnp.inf), offset, axis1, axis2
        )
        ret_edited = ret_edited.at[ret_edited == -jnp.inf].set(-0.0)
        ret = ret.at[ret == ret_edited].set(ret_edited[ret == ret_edited])
    else:
        ret = jnp.diagonal(x, offset, axis1, axis2)
    return ret


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def eigh(
    x: JaxArray, /, *, UPLO: Optional[str] = "L", out: Optional[JaxArray] = None
) -> Tuple[JaxArray]:
    result_tuple = NamedTuple(
        "eigh", [("eigenvalues", JaxArray), ("eigenvectors", JaxArray)]
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(x, UPLO=UPLO)
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def eigvalsh(
    x: JaxArray, /, *, UPLO: Optional[str] = "L", out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.linalg.eigvalsh(x, UPLO=UPLO)


@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
def inner(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.inner(x1, x2)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def inv(
    x: JaxArray,
    /,
    *,
    adjoint: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:

    if jnp.any(jnp.linalg.det(x.astype("float64")) == 0):
        return x
    else:
        if adjoint is False:
            ret = jnp.linalg.inv(x)
            return ret
        else:
            x = jnp.transpose(x)
            ret = jnp.linalg.inv(x)
            return ret


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def matmul(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if transpose_a is True:
        x1 = jnp.transpose(x1)
    if transpose_b is True:
        x2 = jnp.transpose(x2)
    return jnp.matmul(x1, x2)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def matrix_norm(
    x: JaxArray,
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    axis: Optional[Tuple[int, int]] = (-2, -1),
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if not isinstance(axis, tuple):
        axis = tuple(axis)
    return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
def matrix_power(x: JaxArray, n: int, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.linalg.matrix_power(x, n)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def matrix_rank(
    x: JaxArray,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
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
            return jnp.asarray(0).astype(x.dtype)
    elif len(x.shape) > 3:
        if x.shape[-3] == 0 or x.shape[-4] == 0:
            return jnp.asarray(0).astype(x.dtype)
    axis = None
    ret_shape = x.shape[:-2]
    if len(x.shape) == 2:
        singular_values = jnp.linalg.svd(x, compute_uv=False)
    elif len(x.shape) > 2:
        y = x.reshape((-1, *x.shape[-2:]))
        singular_values = jnp.asarray(
            [
                jnp.linalg.svd(split[0], compute_uv=False)
                for split in jnp.split(y, y.shape[0], axis=0)
            ]
        )
        axis = 1
    if len(x.shape) < 2 or len(singular_values.shape) == 0:
        return jnp.array(0, dtype=x.dtype)
    max_values = jnp.max(singular_values, axis=axis)
    if atol is None:
        if rtol is None:
            ret = jnp.sum(singular_values != 0, axis=axis)
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
            ret = jnp.sum(singular_values > atol, axis=axis)
        else:
            tol = jnp.max(atol, max_values * rtol)
            ret = jnp.sum(singular_values > tol, axis=axis)
    if len(ret_shape):
        ret = ret.reshape(ret_shape)
    return ret.astype(x.dtype)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "int",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def matrix_transpose(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.swapaxes(x, -1, -2)


@with_unsupported_dtypes({"0.3.14 and below": ("complex", )}, backend_version)
def outer(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.outer(x1, x2)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
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
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def qr(
    x: JaxArray, /, *, mode: str = "reduced", out: Optional[JaxArray] = None
) -> Tuple[JaxArray, JaxArray]:
    res = namedtuple("qr", ["Q", "R"])
    q, r = jnp.linalg.qr(x, mode=mode)
    return res(q, r)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
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
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def solve(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
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
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
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
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
    backend_version,
)
def svdvals(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.linalg.svd(x, compute_uv=False)


@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
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
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
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


@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
def vecdot(
    x1: JaxArray, x2: JaxArray, /, *, axis: int = -1, out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.tensordot(x1, x2, axes=(axis, axis))


@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
def vector_norm(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if axis is None:
        jnp_normalized_vector = jnp.linalg.norm(jnp.ravel(x), ord, axis, keepdims)
    else:
        jnp_normalized_vector = jnp.linalg.norm(x, ord, axis, keepdims)

    if jnp_normalized_vector.shape == ():
        ret = jnp.expand_dims(jnp_normalized_vector, 0)
    else:
        ret = jnp_normalized_vector
    return ret


# Extra #
# ------#


@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
def diag(
    x: JaxArray,
    /,
    *,
    k: int = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.diag(x, k=k)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
            "complex"
        )
    },
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


@with_unsupported_dtypes({"0.3.14 and below": ("complex",)}, backend_version)
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
