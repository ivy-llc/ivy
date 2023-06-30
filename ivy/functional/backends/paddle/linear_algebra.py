# global

import paddle
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence
from collections import namedtuple


# local
import ivy
from ivy import inf
from ivy.utils.exceptions import IvyNotImplementedException
import ivy.functional.backends.paddle as paddle_backend
from . import backend_version
from ivy.func_wrapper import with_unsupported_device_and_dtypes, with_unsupported_dtypes
from .elementwise import _elementwise_helper

# Array API Standard #
# -------------------#


@with_unsupported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "float16",
                "complex",
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
            return paddle.complex(
                _cross(x1.real(), x2.real(), axisa, axisb, axisc, axis),
                _cross(x1.real(), x2.real(), axisa, axisb, axisc, axis),
            )
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
    {"2.5.0 and below": {"cpu": ("complex",)}},
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
        ret = paddle.linalg.det(x.cast("float32")).cast(x.dtype)
    else:
        ret = paddle.linalg.det(x)
    if x.ndim == 2:
        ret = paddle_backend.squeeze(ret, axis=0)
    return ret


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
            return paddle.complex(
                paddle.diagonal(x.real(), offset=offset, axis1=axis1, axis2=axis2),
                paddle.diagonal(x.imag(), offset=offset, axis1=axis1, axis2=axis2),
            )
        return paddle.diagonal(
            x.cast("float32"), offset=offset, axis1=axis1, axis2=axis2
        ).cast(x.dtype)
    return paddle.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


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


def eigvalsh(
    x: paddle.Tensor,
    /,
    *,
    UPLO: Optional[str] = "L",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.linalg.eigvalsh(x, UPLO=UPLO)


def inner(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
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
    return paddle.inner(x1, x2).squeeze().cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("complex",)}},
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
    ret = paddle.matmul(x1, x2, transpose_x=transpose_a, transpose_y=transpose_b).cast(
        ret_dtype
    )
    # handle case where ret should be 0d.
    if x1.ndim == 1 and x2.ndim == 1:
        ret_dtype = ret.dtype
        if ret_dtype in [paddle.int16]:
            ret = ret.cast(paddle.int32)
        return ret.squeeze().astype(ret_dtype)

    return ret


def matrix_norm(
    x: paddle.Tensor,
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    axis: Optional[Tuple[int, int]] = (-2, -1),
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    axis_ = list(axis)  # paddle.moveaxis doesn't support tuple axes
    if ord == "nuc":
        x = paddle.moveaxis(x, axis_, [-2, -1])
        # backend implementation is used here instead of native implementation
        # because native implementation causes issues when the return should be
        # a scalar which is solved in the backend implementation
        ret = paddle_backend.sum(
            paddle_backend.svd(x)[1],
            axis=-1,
        )
    elif ord == 1:
        ret = paddle_backend.max(
            paddle.sum(paddle_backend.abs(x), axis=axis[0], keepdim=True),
            axis=axis,
            keepdims=keepdims,
        )
    elif ord == -1:
        ret = paddle_backend.min(
            paddle.sum(paddle_backend.abs(x), axis=axis[0], keepdim=True),
            axis=axis,
            keepdims=keepdims,
        )
    elif ord == 2:
        x = paddle.moveaxis(x, axis_, [-2, -1])
        ret = paddle_backend.max(
            paddle_backend.svd(x)[1],
            axis=-1,
        )
    elif ord == -2:
        x = paddle.moveaxis(x, axis_, [-2, -1])
        ret = paddle_backend.min(
            paddle_backend.svd(x)[1],
            axis=-1,
        )
    elif ord == float("inf"):
        ret = paddle_backend.max(
            paddle.sum(paddle.abs(x), axis=axis[1], keepdim=True),
            axis=axis,
            keepdims=keepdims,
        )
    elif ord == float("-inf"):
        ret = paddle_backend.min(
            paddle.sum(paddle.abs(x), axis=axis[1], keepdim=True),
            axis=axis,
            keepdims=keepdims,
        )
    else:
        ret = paddle.linalg.norm(x, p=ord, axis=axis, keepdim=keepdims)
    if x.ndim == 2 and not keepdims:
        ret = paddle.squeeze(ret)
    elif keepdims and ord in ["nuc", -2, 2]:
        # only these norms because the use of SVD
        for dim in axis:
            # although expand_dims support tuple axes, we have to loop
            # over the axes because it faces problems when the input is a scalar
            ret = paddle_backend.expand_dims(ret, axis=dim % x.ndim)
    return ret


def eig(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> Tuple[paddle.Tensor]:
    result_tuple = NamedTuple(
        "eig", [("eigenvalues", paddle.Tensor), ("eigenvectors", paddle.Tensor)]
    )
    eigenvalues, eigenvectors = paddle.linalg.eig(x)
    return result_tuple(eigenvalues, eigenvectors)


def matrix_power(
    x: paddle.Tensor, n: int, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.linalg.matrix_power(x, n)


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("complex",)}},
    backend_version,
)
def matrix_rank(
    x: paddle.Tensor,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    hermitian: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if (x.ndim < 2) or (0 in x.shape):
        return paddle.to_tensor(0).squeeze().astype(x.dtype)
    # we don't use the native matrix_rank function because the behaviour of the
    # tolerance argument is difficult to unify

    if hermitian:
        svd_values = paddle_backend.abs(paddle_backend.eigvalsh(x))
    else:
        svd_values = paddle_backend.svd(x)[1]
    sigma = paddle_backend.max(svd_values, axis=-1, keepdims=False)
    atol = (
        atol if atol is not None else ivy.finfo(x.dtype).eps * max(x.shape[-2:]) * sigma
    )
    rtol = rtol if rtol is not None else 0.0
    tol = paddle_backend.maximum(atol, paddle_backend.multiply(rtol, sigma))
    # make sure it's broadcastable again with svd_values
    tol = paddle_backend.expand_dims(tol, axis=-1)
    ret = paddle.count_nonzero(paddle_backend.greater(svd_values, tol), axis=-1)
    if x.ndim == 2 and tol.ndim < 2:
        # to fix the output shape when input is unbatched
        # and tol is batched
        ret = paddle_backend.squeeze(ret, axis=None)
    return ret


def matrix_transpose(
    x: paddle.Tensor,
    /,
    *,
    perm: Optional[Union[Tuple[int], List[int]]] = None,
    conjugate: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if conjugate:
        x = paddle.conj(x)
    perm = list(range(x.ndim))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    if x.dtype in [paddle.int8, paddle.int16, paddle.uint8]:
        return paddle.transpose(x.cast("float32"), perm=perm).cast(x.dtype)
    return paddle.transpose(x, perm=perm)


def outer(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
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
    return paddle.outer(x1, x2).cast(ret_dtype)


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


def slogdet(
    x: paddle.Tensor,
    /,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    results = NamedTuple(
        "slogdet", [("sign", paddle.Tensor), ("logabsdet", paddle.Tensor)]
    )
    sign, logabsdet = paddle.linalg.slogdet(x)
    if x.ndim == 2:
        sign, logabsdet = paddle_backend.squeeze(sign, axis=0), paddle_backend.squeeze(
            logabsdet, axis=0
        )
    return results(sign, logabsdet)


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
    {"2.5.0 and below": {"cpu": ("complex",)}},
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
        ret = paddle.linalg.svd(x.cast("float32"), full_matrices=full_matrices)
        ret = tuple(r.cast(x.dtype) for r in ret)
    else:
        ret = paddle.linalg.svd(x, full_matrices=full_matrices)
    if compute_uv:
        results = namedtuple("svd", "U S Vh")
        return results(*ret)
    else:
        results = namedtuple("svd", "S")
        return results(ret[1])


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("complex64", "complex128")}},
    backend_version,
)
def svdvals(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle_backend.svd(x)[1]


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
    ret = paddle.tensordot(x1, x2, axes=axes)
    return ret.squeeze().cast(ret_dtype) if x1.ndim == axes else ret.cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "int8",
                "int16",
                "unsigned",
                "float16",
                "complex",
                "bool",
            )
        }
    },
    backend_version,
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
    ret = paddle.trace(x, offset=offset, axis1=axis1, axis2=axis2)
    return ret.squeeze() if x.ndim <= 2 else ret


def vecdot(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    axis: int = -1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    axes = [axis % x1.ndim]

    return paddle_backend.tensordot(x1, x2, axes=axes)


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
    if dtype in ["complex64", "complex128"]:
        dtype = "float" + str(ivy.dtype_bits(dtype) // 2)
    if x.ndim == 0:
        x = paddle_backend.expand_dims(x, axis=0)
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
            x = paddle.abs(x)
            ret = paddle.norm(x, p=ord, axis=axis, keepdim=keepdims).astype(dtype)
        else:
            ret = paddle.norm(
                x.cast("float32"), p=ord, axis=axis, keepdim=keepdims
            ).astype(dtype)
    else:
        ret = paddle.norm(x, p=ord, axis=axis, keepdim=keepdims).astype(dtype)
    if ret_scalar or (x.ndim == 1 and not keepdims):
        ret = paddle_backend.squeeze(ret, axis=axis)
    return ret


# Extra #
# ----- #


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
            return paddle.complex(
                paddle.diag(x.real(), offset=k), paddle.diag(x.imag(), offset=k)
            )
        return paddle.diag(x.cast("float32"), offset=k).cast(x.dtype)
    return paddle.diag(x, offset=k)


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("uint8", "int8", "int16")}},
    backend_version,
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


@with_unsupported_dtypes(
    {"2.5.0 and below": ("unsigned", "int8", "int16", "float16")},
    backend_version,
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
