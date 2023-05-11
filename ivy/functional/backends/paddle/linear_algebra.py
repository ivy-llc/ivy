# global

import paddle
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence

from collections import namedtuple


# local
import ivy
from ivy import inf
from ivy.utils.exceptions import IvyNotImplementedException
from . import backend_version
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from .elementwise import _elementwise_helper

# Array API Standard #
# -------------------#


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "bfloat16",
                "float16",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def cholesky(
    x: paddle.Tensor, /, *, upper: bool = False, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.linalg.cholesky(x, upper=upper)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def cross(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    def _cross(x1, x2, axisa, axisb, axisc, axis):
        if axis is not None:
            return paddle.cross(x1, x2, axis=axis)
        x1 = paddle.moveaxis(x1, axisa, 1)
        x2 = paddle.moveaxis(x2, axisb, 1)
        ret = paddle.cross(x1, x2)
        return paddle.moveaxis(ret, 1, axisc)

    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x1):
            return _cross(
                x1.real(), x2.real(), axisa, axisb, axisc, axis
            ) + 1j * _cross(x1.real(), x2.real(), axisa, axisb, axisc, axis)
        return _cross(
            x1.cast("float32"),
            x2.cast("float32"),
            axisa,
            axisb,
            axisc,
            axis,
        ).cast(ret_dtype)
    return _cross(x1, x2, axisa, axisb, axisc, axis)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "complex64", "complex128")}},
    backend_version,
)
def det(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        return paddle.linalg.det(x.cast("float32")).squeeze().cast(x.dtype)
    return paddle.linalg.det(x).squeeze()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def diagonal(
    x: paddle.Tensor,
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
    ]:
        if paddle.is_complex(x):
            return paddle.diagonal(
                x.real(), offset=offset, axis1=axis1, axis2=axis2
            ) + 1j * paddle.diagonal(x.imag(), offset=offset, axis1=axis1, axis2=axis2)
        return paddle.diagonal(
            x.cast("float32"), offset=offset, axis1=axis1, axis2=axis2
        ).cast(x.dtype)
    return paddle.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def eigh(
    x: paddle.Tensor,
    /,
    *,
    UPLO: Optional[str] = "L",
    out: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor]:
    result_tuple = NamedTuple(
        "eigh", [("eigenvalues", paddle.Tensor), ("eigenvectors", paddle.Tensor)]
    )
    eigenvalues, eigenvectors = paddle.linalg.eigh(x, UPLO=UPLO)
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def eigvalsh(
    x: paddle.Tensor,
    /,
    *,
    UPLO: Optional[str] = "L",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.linalg.eigvalsh(x, UPLO=UPLO)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def inner(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")
    return paddle.inner(x1, x2).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "complex64", "complex128")}},
    backend_version,
)
def inv(
    x: paddle.Tensor,
    /,
    *,
    adjoint: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = x.dtype
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        x = x.cast("float32")
    if adjoint:
        x = paddle.moveaxis(x, -2, -1).conj()
    return paddle.inverse(x).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def matmul(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret_dtype = x1.dtype
    if x1.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")

    if adjoint_a:
        x1 = paddle.moveaxis(x1, -2, -1).conj()
    if adjoint_b:
        x2 = paddle.moveaxis(x2, -2, -1).conj()
    return paddle.matmul(x1, x2, transpose_x=transpose_a, transpose_y=transpose_b).cast(
        ret_dtype
    )


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def matrix_norm(
    x: paddle.Tensor,
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    axis: Optional[Tuple[int, int]] = (-2, -1),
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    _expand_dims = False
    if x.ndim == 2:
        x = paddle.unsqueeze(x, axis=0)
        _expand_dims = True

    if ord == -float("inf"):
        ret = paddle.amin(
            paddle.sum(paddle.abs(x), axis=axis[1], keepdim=True),
            axis=axis,
            keepdim=keepdims,
        )

    elif ord == -1:
        ret = paddle.amin(
            paddle.sum(paddle.abs(x), axis=axis[0], keepdim=True),
            axis=axis,
            keepdim=keepdims,
        )
    elif ord == -2:
        ret = paddle.amin(paddle.linalg.svd(x)[1], axis=axis[1], keepdim=keepdims)
        if keepdims:
            ret = ret.unsqueeze(-1)
    elif ord == "nuc":
        if x.size == 0:
            ret = x
        else:
            ret = paddle.sum(paddle.linalg.svd(x)[1], axis=-1, keepdim=keepdims)
            if keepdims:
                ret = ret.unsqueeze(-1)
    elif ord == "fro":
        ret = paddle.linalg.norm(x, p=ord, axis=axis, keepdim=keepdims)
    elif ord == float("inf"):
        ret = paddle.amax(
            paddle.sum(paddle.abs(x), axis=axis[1], keepdim=True),
            axis=axis,
            keepdim=keepdims,
        )

    elif ord == 1:
        ret = paddle.amax(
            paddle.sum(paddle.abs(x), axis=axis[0], keepdim=True),
            axis=axis,
            keepdim=keepdims,
        )
    elif ord == 2:
        ret = paddle.amax(paddle.linalg.svd(x)[1], axis=axis[1], keepdim=keepdims)
        if keepdims:
            ret = ret.unsqueeze(-1)

    if _expand_dims:
        ret = paddle.squeeze(ret, axis=0)
    return ret


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def eig(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> Tuple[paddle.Tensor]:
    result_tuple = NamedTuple(
        "eig", [("eigenvalues", paddle.Tensor), ("eigenvectors", paddle.Tensor)]
    )
    eigenvalues, eigenvectors = paddle.linalg.eig(x)
    return result_tuple(eigenvalues, eigenvectors)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def matrix_power(
    x: paddle.Tensor, n: int, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.linalg.matrix_power(x, n)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "complex64", "complex128")}},
    backend_version,
)
def matrix_rank(
    x: paddle.Tensor,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.ndim < 2:
        return paddle.to_tensor(0, dtype=x.dtype)
    else:
        atol = atol if atol is not None else 0
        rtol = rtol if rtol is not None else 0
        with ivy.ArrayMode(False):
            svd_values = svd(x)[1]
            sigma = ivy.max(svd_values)
            tol = ivy.maximum(atol, rtol * sigma)
        if x.dtype in [
            paddle.int8,
            paddle.int16,
            paddle.int32,
            paddle.int64,
            paddle.uint8,
            paddle.float16,
            paddle.bool,
        ]:
            return paddle.linalg.matrix_rank(x.cast("float32"), tol=tol).cast(x.dtype)
        return paddle.linalg.matrix_rank(x, tol=tol).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def matrix_transpose(
    x: paddle.Tensor,
    /,
    *,
    perm: Union[Tuple[List[int], List[int]]] = None,
    conjugate: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    perm = list(range(x.ndim))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    if x.dtype in [paddle.int8, paddle.int16, paddle.uint8]:
        return paddle.transpose(x.cast("float32"), perm=perm).cast(x.dtype)
    return paddle.transpose(x, perm=perm)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def outer(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")
    return paddle.outer(x1, x2).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def pinv(
    x: paddle.Tensor,
    /,
    *,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if rtol is None:
        return paddle.linalg.pinv(x)
    return paddle.linalg.pinv(x, rcond=rtol)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def tensorsolve(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    axes: Union[int, Tuple[List[int], List[int]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    # Implemented as a composite function in ivy.functional.ivy.linear_algebra
    raise IvyNotImplementedException()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def qr(
    x: paddle.Tensor,
    /,
    *,
    mode: str = "reduced",
    out: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    res = namedtuple("qr", ["Q", "R"])
    q, r = paddle.linalg.qr(x, mode=mode)
    return res(q, r)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def slogdet(
    x: paddle.Tensor,
    /,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    results = NamedTuple(
        "slogdet", [("sign", paddle.Tensor), ("logabsdet", paddle.Tensor)]
    )
    sign, logabsdet = paddle.linalg.slogdet(x)
    return results(sign, logabsdet)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def solve(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    adjoint: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if adjoint:
        x1 = paddle.moveaxis(x1, -2, -1).conj()
    expanded_last = False
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if len(x2.shape) <= 1:
        if x2.shape[-1] == x1.shape[-1]:
            expanded_last = True
            x2 = paddle.unsqueeze(x2, axis=1)
    for i in range(len(x1.shape) - 2):
        x2 = paddle.unsqueeze(x2, axis=0)
    ret = paddle.linalg.solve(x1, x2)
    if expanded_last:
        ret = paddle.squeeze(ret, axis=-1)
    return ret


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "complex64", "complex128")}},
    backend_version,
)
def svd(
    x: paddle.Tensor, /, *, full_matrices: bool = True, compute_uv: bool = True
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        ret = paddle.linalg.svd(x.cast("float32"), full_matrices=full_matrices).cast(
            x.dype
        )
    else:
        ret = paddle.linalg.svd(x, full_matrices=full_matrices)
    if compute_uv:
        results = namedtuple("svd", "U S Vh")
        return results(*ret)
    else:
        results = namedtuple("svd", "S")
        return results(ret[1])


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "complex64", "complex128")}},
    backend_version,
)
def svdvals(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        return svd(x)[1]


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def tensordot(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret_dtype = x1.dtype
    if x1.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")
    return paddle.tensordot(x1, x2, axes=axes).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def trace(
    x: paddle.Tensor,
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.trace(x, offset=offset, axis1=axis1, axis2=axis2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def vecdot(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    axis: int = -1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    axes = [axis % x1.ndim]
    with ivy.ArrayMode(False):
        return tensordot(x1, x2, axes=axes)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def vector_norm(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: Optional[bool] = False,
    ord: Optional[Union[int, float, Literal[inf, -inf]]] = 2,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_scalar = False
    dtype = dtype if dtype is not None else x.dtype
    if x.ndim == 0:
        x = ivy.to_native(ivy.expand_dims(x, axis=0))
        ret_scalar = True

    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            x = ivy.to_native(ivy.abs(x))
            ret = paddle.norm(x, p=ord, axis=axis, keepdim=keepdims).astype(dtype)
        else:
            ret = paddle.norm(
                x.cast("float32"), p=ord, axis=axis, keepdim=keepdims
            ).astype(dtype)
    else:
        ret = paddle.norm(x, p=ord, axis=axis, keepdim=keepdims).astype(dtype)
    return ivy.squeeze(ret, axis=-1) if ret_scalar else ret


# Extra #
# ----- #


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def diag(
    x: paddle.Tensor,
    /,
    *,
    k: int = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            return paddle.diag(x.real(), offset=k) + 1j * paddle.diag(
                x.imag(), offset=k
            )
        return paddle.diag(x.cast("float32"), offset=k).cast(x.dtype)
    return paddle.diag(x, offset=k)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def vander(
    x: paddle.Tensor,
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    N = ivy.default(N, x.shape[-1])
    start, stop, step = N - 1, -1, -1
    if increasing:
        start, stop, step = 0, N, 1
    return paddle.pow(
        paddle.moveaxis(paddle.unsqueeze(x, 0), 0, 1),
        paddle.arange(start, stop, step, dtype=x.dtype),
    )


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def vector_to_skew_symmetric_matrix(
    vector: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    batch_shape = vector.shape[:-1]
    # BS x 3 x 1
    vector_expanded = paddle.unsqueeze(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = paddle.zeros(batch_shape + [1, 1], dtype=vector.dtype)
    # BS x 1 x 3
    row1 = paddle.concat((zs, -a3s, a2s), -1)
    row2 = paddle.concat((a3s, zs, -a1s), -1)
    row3 = paddle.concat((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return paddle.concat((row1, row2, row3), -2)
