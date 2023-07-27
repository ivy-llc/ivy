# global

import torch
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence

from collections import namedtuple


# local
import ivy
from ivy import inf
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from . import backend_version
from .elementwise import _cast_for_unary_op


# Array API Standard #
# -------------------#


@with_unsupported_dtypes(
    {"2.0.1 and below": ("bfloat16", "float16", "complex")},
    backend_version,
)
def cholesky(
    x: torch.Tensor, /, *, upper: bool = False, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if not upper:
        return torch.linalg.cholesky(x, out=out)
    else:
        ret = torch.transpose(
            torch.linalg.cholesky(
                torch.transpose(x, dim0=len(x.shape) - 1, dim1=len(x.shape) - 2)
            ),
            dim0=len(x.shape) - 1,
            dim1=len(x.shape) - 2,
        )
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret


cholesky.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
def cross(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = -1
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)

    if axis is not None:
        return torch.linalg.cross(input=x1, other=x2, dim=axis)
    x1 = torch.transpose(x1, axisa, 1)
    x2 = torch.transpose(x2, axisb, 1)
    return torch.transpose(
        torch.linalg.cross(input=x1, other=x2, out=out), dim0=axisc, dim1=1
    )


cross.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def det(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.linalg.det(x, out=out)


det.support_native_out = True


def diagonal(
    x: torch.Tensor,
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def eigh(
    x: torch.Tensor, /, *, UPLO: str = "L", out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor]:
    result_tuple = NamedTuple(
        "eigh", [("eigenvalues", torch.Tensor), ("eigenvectors", torch.Tensor)]
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(x, UPLO=UPLO, out=out)
    return result_tuple(eigenvalues, eigenvectors)


eigh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def eigvalsh(
    x: torch.Tensor, /, *, UPLO: str = "L", out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.linalg.eigvalsh(x, UPLO=UPLO, out=out)


eigvalsh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def inner(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret_dtype = x1.dtype
    if ivy.is_int_dtype(x1):
        # https://github.com/pytorch/pytorch/issues/103366
        x1 = x1.long()
        x2 = x2.long()
        ret = torch.inner(x1, x2).type(ret_dtype)
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret
    return torch.inner(x1, x2, out=out)


inner.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def inv(
    x: torch.Tensor,
    /,
    *,
    adjoint: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if adjoint:
        if x.dim() < 2:
            raise ValueError("Input must be at least 2D")
        x_adj = x.transpose(-2, -1).conj()
        ret = torch.linalg.inv(x_adj)
    else:
        ret = torch.linalg.inv(x)

    if ivy.exists(out):
        return ivy.inplace_update(out, ret)

    return ret


inv.support_native_out = True


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "bool")}, backend_version
)
def matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # torch does not support inplace matmul (same storage in out=)
    # https://github.com/pytorch/pytorch/issues/58742
    # https://github.com/pytorch/pytorch/issues/48900
    if out is x1 or out is x2:
        out = None
    if transpose_a:
        x1 = torch.swapaxes(x1, -1, -2)
    if transpose_b:
        x2 = torch.swapaxes(x2, -1, -2)
    if adjoint_a:
        x1 = torch.adjoint(x1)
    if adjoint_b:
        x2 = torch.adjoint(x2)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.matmul(x1, x2, out=out)


matmul.support_native_out = True


@with_supported_dtypes({"2.0.1 and below": ("float", "complex")}, backend_version)
def matrix_norm(
    x: torch.Tensor,
    /,
    *,
    ord: Union[int, float, Literal[inf, -inf, "fro", "nuc"]] = "fro",
    axis: Tuple[int, int] = (-2, -1),
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.linalg.matrix_norm(x, ord=ord, dim=axis, keepdim=keepdims, out=out)


matrix_norm.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def eig(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor]:
    result_tuple = NamedTuple(
        "eig", [("eigenvalues", torch.Tensor), ("eigenvectors", torch.Tensor)]
    )
    eigenvalues, eigenvectors = torch.linalg.eig(x, out=out)
    return result_tuple(eigenvalues, eigenvectors)


eig.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def matrix_power(
    x: torch.Tensor, n: int, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.linalg.matrix_power(x, n, out=out)


matrix_power.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def matrix_rank(
    x: torch.Tensor,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    hermitian: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if (x.ndim < 2) or (0 in x.shape):
        return torch.tensor(0, dtype=torch.int64)
    # we don't use the native matrix_rank function because the behaviour of the
    # tolerance argument is difficult to unify
    # return torch.linalg.matrix_rank(
    #     x, atol=atol, rtol=rtol, hermitian=hermitian, out=out
    # )
    if hermitian:
        svd_values = torch.abs(torch.linalg.eigvalsh(x))
    else:
        svd_values = torch.linalg.svdvals(x)
    sigma = torch.max(svd_values, axis=-1, keepdim=False)[0]
    atol = (
        atol
        if atol is not None
        else torch.finfo(x.dtype).eps * max(x.shape[-2:]) * sigma
    )
    rtol = rtol if rtol is not None else 0.0
    atol = _cast_for_unary_op(atol)
    rtol = _cast_for_unary_op(rtol)
    tol = torch.maximum(atol, rtol * sigma)
    # make sure it's broadcastable again with svd_values
    tol = torch.unsqueeze(tol, dim=-1)
    ret = torch.count_nonzero(svd_values > tol, dim=-1)
    return ret


matrix_rank.support_native_out = True


def matrix_transpose(
    x: torch.Tensor, /, *, conjugate: bool = False, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if conjugate:
        x = torch.conj(x)
    return torch.swapaxes(x, -1, -2)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def outer(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.outer(x1, x2, out=out)


outer.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def pinv(
    x: torch.Tensor,
    /,
    *,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if rtol is None:
        return torch.linalg.pinv(x, out=out)
    return torch.linalg.pinv(x, rtol, out=out)


pinv.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def tensorsolve(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    axes: Optional[Union[int, Tuple[List[int], List[int]]]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.linalg.tensorsolve(x1, x2, dims=axes)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def qr(
    x: torch.Tensor,
    /,
    *,
    mode: str = "reduced",
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    res = namedtuple("qr", ["Q", "R"])
    if mode == "reduced":
        q, r = torch.qr(x, some=True, out=out)
        ret = res(q, r)
    elif mode == "complete":
        q, r = torch.qr(x, some=False, out=out)
        ret = res(q, r)
    else:
        raise ivy.utils.exceptions.IvyException(
            "Only 'reduced' and 'complete' qr modes are allowed for the torch backend."
        )
    return ret


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def slogdet(
    x: torch.Tensor,
    /,
) -> Tuple[torch.Tensor, torch.Tensor]:
    results = NamedTuple(
        "slogdet", [("sign", torch.Tensor), ("logabsdet", torch.Tensor)]
    )
    sign, logabsdet = torch.linalg.slogdet(x)
    return results(sign, logabsdet)


slogdet.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def solve(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    adjoint: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if adjoint:
        x1 = torch.adjoint(x1)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    expanded_last = False
    if len(x2.shape) <= 1:
        if x2.shape[-1] == x1.shape[-1]:
            expanded_last = True
            x2 = torch.unsqueeze(x2, dim=1)

    is_empty_x1 = x1.nelement() == 0
    is_empty_x2 = x2.nelement() == 0
    if is_empty_x1 or is_empty_x2:
        for i in range(len(x1.shape) - 2):
            x2 = torch.unsqueeze(x2, dim=0)
        output_shape = list(torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
        output_shape.append(x2.shape[-2])
        output_shape.append(x2.shape[-1])
        ret = torch.Tensor([])
        ret = torch.reshape(ret, output_shape)
    else:
        ret = torch.linalg.solve(x1, x2)

    if expanded_last:
        ret = torch.squeeze(ret, dim=-1)
    return ret


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def svd(
    x: torch.Tensor, /, *, full_matrices: bool = True, compute_uv: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if compute_uv:
        results = namedtuple("svd", "U S Vh")

        U, D, VT = torch.linalg.svd(x, full_matrices=full_matrices)
        return results(U, D, VT)
    else:
        results = namedtuple("svd", "S")
        svd = torch.linalg.svd(x, full_matrices=full_matrices)
        # torch.linalg.svd returns a tuple with U, S, and Vh
        D = svd[1]
        return results(D)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def svdvals(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.linalg.svdvals(x, out=out)


svdvals.support_native_out = True


# ToDo: re-add int32 support once
# (https://github.com/pytorch/pytorch/issues/84530) is fixed
@with_unsupported_dtypes({"2.0.1 and below": ("int32",)}, backend_version)
def tensordot(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # find the type to promote to
    dtype = ivy.as_native_dtype(ivy.promote_types(x1.dtype, x2.dtype))
    # type conversion to one that torch.tensordot can work with
    x1, x2 = x1.type(torch.float32), x2.type(torch.float32)

    # handle tensordot for axes==0
    # otherwise call with axes
    if axes == 0:
        ret = (x1.reshape(x1.size() + (1,) * x2.dim()) * x2).type(dtype)
    else:
        ret = torch.tensordot(x1, x2, dims=axes).type(dtype)
    return ret


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def trace(
    x: torch.Tensor,
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if len(x) == 0:
        return ivy.array([])
    ret = torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)
    ret = torch.sum(ret, dim=-1)
    return ret


def vecdot(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    axis: int = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(ivy.promote_types(x1.dtype, x2.dtype))
    if dtype != "float64":
        x1, x2 = x1.to(dtype=torch.float32), x2.to(dtype=torch.float32)
    if ivy.exists(out):
        if ivy.as_ivy_dtype(out.dtype) == ivy.as_ivy_dtype(x1.dtype):
            return torch.tensordot(x1, x2, dims=([axis], [axis]), out=out)
        return ivy.inplace_update(
            out, torch.tensordot(x1, x2, dims=([axis], [axis])).to(out.dtype)
        )
    return torch.tensordot(x1, x2, dims=([axis], [axis])).to(dtype)


vecdot.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("integer",)}, backend_version)
def vector_norm(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # TODO: remove the as_native_dtype call once there are wrappers that handle dtype
    #  conversion automatically in the backends
    dtype = ivy.as_native_dtype(dtype)
    if dtype and x.dtype != dtype:
        x = x.type(dtype)
    return torch.linalg.vector_norm(x, ord, axis, keepdims, out=out)


vector_norm.support_native_out = True


# Extra #
# ----- #


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def diag(
    x: torch.Tensor,
    /,
    *,
    k: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.tensor:
    return torch.diag(x, diagonal=k)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def vander(
    x: torch.tensor,
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    # torch.vander hasn't been used as it produces 0 gradients
    N = ivy.default(N, x.shape[-1])
    start, stop, step = N - 1, -1, -1
    if increasing:
        start, stop, step = 0, N, 1
    ret = torch.pow(
        torch.transpose(torch.unsqueeze(x, 0), 0, 1),
        torch.arange(start, stop, step),
        out=out,
    )
    if ret.dtype != x.dtype:
        return ret.to(x.dtype)
    return ret


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "complex",
            "unsigned",
        )
    },
    backend_version,
)
def vector_to_skew_symmetric_matrix(
    vector: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = torch.unsqueeze(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = torch.zeros(batch_shape + [1, 1], device=vector.device, dtype=vector.dtype)
    # BS x 1 x 3
    row1 = torch.cat((zs, -a3s, a2s), -1)
    row2 = torch.cat((a3s, zs, -a1s), -1)
    row3 = torch.cat((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    return torch.cat((row1, row2, row3), -2, out=out)


vector_to_skew_symmetric_matrix.support_native_out = True
