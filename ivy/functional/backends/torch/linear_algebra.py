# global

import torch
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence

from collections import namedtuple


# local
import ivy
from ivy import inf
from ivy.func_wrapper import with_unsupported_dtypes
from . import version


# Array API Standard #
# -------------------#


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
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


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, version)
def cross(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    if axis is None:
        axis = -1
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)

    if axis:
        return torch.linalg.cross(input=x1, other=x2, dim=axis)
    x1 = torch.transpose(x1, axisa, 1)
    x2 = torch.transpose(x2, axisb, 1)
    return torch.transpose(
        torch.linalg.cross(input=x1, other=x2, out=out), dim0=axisc, dim1=1
    )


cross.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
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


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
def eigh(
    x: torch.Tensor, /, *, UPLO: Optional[str] = "L", out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor]:
    result_tuple = NamedTuple(
        "eigh", [("eigenvalues", torch.Tensor), ("eigenvectors", torch.Tensor)]
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(x, UPLO=UPLO, out=out)
    return result_tuple(eigenvalues, eigenvectors)


eigh.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
def eigvalsh(
    x: torch.Tensor, /, *, UPLO: Optional[str] = "L", out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.linalg.eigvalsh(x, UPLO=UPLO, out=out)


eigvalsh.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("int8",)}, version)
def inner(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.inner(x1, x2, out=out)


inner.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
def inv(
    x: torch.Tensor,
    /,
    *,
    adjoint: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if torch.linalg.det == 0:
        ret = x
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
    else:
        if adjoint is False:
            ret = torch.inverse(x, out=out)
            return ret
        else:
            x = torch.t(x)
            ret = torch.inverse(x, out=out)
            if ivy.exists(out):
                return ivy.inplace_update(out, ret)
            return ret


inv.support_native_out = True


def matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    if transpose_a is True:
        x1 = torch.t(x1)
    if transpose_b is True:
        x2 = torch.t(x2)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.matmul(x1, x2, out=out)


matmul.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
def matrix_norm(
    x: torch.Tensor,
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    axis: Optional[Tuple[int, int]] = (-2, -1),
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.linalg.matrix_norm(x, ord=ord, dim=axis, keepdim=keepdims, out=out)


matrix_norm.support_native_out = True


def matrix_power(
    x: torch.Tensor, n: int, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.linalg.matrix_power(x, n, out=out)


matrix_power.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
def matrix_rank(
    x: torch.Tensor,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if len(x.shape) < 2:
        ret = torch.tensor(0)
    else:
        ret = torch.linalg.matrix_rank(x, atol=atol, rtol=rtol, out=out)
    return ret.to(dtype=x.dtype)


matrix_rank.support_native_out = True


def matrix_transpose(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.swapaxes(x, -1, -2)


def outer(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.outer(x1, x2, out=out)


outer.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
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


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
def qr(
    x: torch.Tensor, mode: str = "reduced", out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    res = namedtuple("qr", ["Q", "R"])
    if mode == "reduced":
        q, r = torch.qr(x, some=True, out=out)
        ret = res(q, r)
    elif mode == "complete":
        q, r = torch.qr(x, some=False, out=out)
        ret = res(q, r)
    else:
        raise ivy.exceptions.IvyException(
            "Only 'reduced' and 'complete' qr modes are allowed for the torch backend."
        )
    return ret


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
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


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
def solve(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
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


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, version)
def svdvals(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.linalg.svdvals(x, out=out)


svdvals.support_native_out = True


# ToDo: re-add int32 support once
# (https://github.com/pytorch/pytorch/issues/84530) is fixed
@with_unsupported_dtypes({"1.11.0 and below": ("int32",)}, version)
def tensordot(
    x1: torch.Tensor,
    x2: torch.Tensor,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    *,
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


def trace(
    x: torch.Tensor,
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ret = torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)
    ret = torch.sum(ret)
    return ret


trace.unsupported_dtypes = ("float16", "bfloat16")


def vecdot(
    x1: torch.Tensor,
    x2: torch.Tensor,
    axis: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(ivy.promote_types(x1.dtype, x2.dtype))
    if dtype != "float64":
        x1, x2 = x1.to(dtype=torch.float32), x2.to(dtype=torch.float32)
    return torch.tensordot(x1, x2, dims=([axis], [axis]), out=out).to(dtype=dtype)


vecdot.support_native_out = True


def vector_norm(
    x: torch.Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    py_normalized_vector = torch.linalg.vector_norm(x, ord, axis, keepdims, out=out)
    if py_normalized_vector.shape == ():
        ret = torch.unsqueeze(py_normalized_vector, 0)
    else:
        ret = py_normalized_vector
    return ret


vector_norm.support_native_out = True


# Extra #
# ----- #


def diag(
    x: torch.Tensor,
    /,
    *,
    offset: int = 0,
    padding_value: float = 0,
    align: str = "RIGHT_LEFT",
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
):
    if len(x.shape) == 1 and offset == 0 and num_rows <= 1 and num_cols <= 1:
        return x

    # This is used as part of Tensorflow's shape calculation
    # See their source code to see what they're doing with it
    lower_diag_index = offset
    upper_diag_index = lower_diag_index

    x_shape = x.shape
    x_rank = len(x_shape)

    num_diags = upper_diag_index - lower_diag_index + 1
    max_diag_len = x_shape[x_rank - 1]

    min_num_rows = max_diag_len - min(upper_diag_index, 0)
    min_num_cols = max_diag_len + max(lower_diag_index, 0)

    if num_rows == -1 and num_cols == -1:
        num_rows = max(min_num_rows, min_num_cols)
        num_cols = num_rows
    elif num_rows == -1:
        num_rows = min_num_rows
    elif num_cols == -1:
        num_cols = min_num_cols

    output_shape = list(x_shape)
    if num_diags == 1:
        output_shape[x_rank - 1] = num_rows
        output_shape.append(num_cols)
    else:
        output_shape[x_rank - 2] = num_rows
        output_shape[x_rank - 1] = num_cols

    output_array = torch.full(tuple(output_shape), padding_value, dtype=x.dtype)
    output_array = output_array.to(x.dtype)

    diag_len = max(min(num_rows, num_cols) - abs(offset) + 1, 1)

    if len(x) < diag_len:
        x = torch.tensor(
            list(x) + [padding_value] * max((diag_len - len(x), 0)), dtype=x.dtype
        )

    temp = x - torch.full(x.shape, padding_value).type(x.dtype)
    diagonal_to_add = torch.diag(temp, diagonal=offset).type(x.dtype)

    diagonal_to_add = diagonal_to_add[tuple(slice(0, n) for n in output_array.shape)]
    diagonal_to_add = diagonal_to_add.to(x.dtype)

    output_array += torch.nn.functional.pad(
        diagonal_to_add,
        (
            0,
            max([output_array.shape[1] - diagonal_to_add.shape[1], 0]),
            0,
            max([output_array.shape[0] - diagonal_to_add.shape[0], 0]),
        ),
        "constant",
        0,
    ).type(x.dtype)
    return output_array.type(x.dtype)


def vander(
    x: torch.tensor,
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.vander(x, N=N, increasing=increasing)


vander.unsupported_dtypes = ("bfloat16", "float16")


def vector_to_skew_symmetric_matrix(
    vector: torch.Tensor, *, out: Optional[torch.Tensor] = None
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
