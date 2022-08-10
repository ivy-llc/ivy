# global
import torch
from typing import Union, Optional, Tuple, Literal, List, NamedTuple
from collections import namedtuple

# local
import ivy
from ivy import inf


# Array API Standard #
# -------------------#


def cholesky(
    x: torch.Tensor, upper: bool = False, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if not upper:
        ret = torch.linalg.cholesky(x, out=out)
    else:
        ret = torch.transpose(
            torch.linalg.cholesky(
                torch.transpose(x, dim0=len(x.shape) - 1, dim1=len(x.shape) - 2),
                out=out,
            ),
            dim0=len(x.shape) - 1,
            dim1=len(x.shape) - 2,
        )
    return ret


cholesky.unsupported_dtypes = ("float16",)


cholesky.support_native_out = True


def cross(
    x1: torch.Tensor,
    x2: torch.Tensor,
    axis: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = -1
    promote_type = torch.promote_types(x1.dtype, x2.dtype)
    x1 = x1.type(promote_type)
    x2 = x2.type(promote_type)
    return torch.linalg.cross(input=x1, other=x2, dim=axis, out=out)


cross.unsupported_dtypes = ("float16",)
cross.support_native_out = True


def det(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.linalg.det(x, out=out)


det.support_native_out = True

det.unsupported_dtypes = ("float16",)


def diagonal(
    x: torch.Tensor,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)


def eigh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.linalg.eigh(x, out=out)


eigh.unsupported_dtypes = ("float16",)

eigh.support_native_out = True


def eigvalsh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.linalg.eigvalsh(x, out=out)


eigvalsh.unsupported_dtypes = ("float16",)

eigvalsh.support_native_out = True


def inv(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.inverse(x, out=out)


inv.unsupported_dtypes = ("float16",)

inv.support_native_out = True


def matmul(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    dtype_from = torch.promote_types(x1.dtype, x2.dtype)
    x1 = x1.type(dtype_from)
    x2 = x2.type(dtype_from)
    return torch.matmul(x1, x2, out=out).type(dtype_from)


matmul.support_native_out = True


def matrix_norm(
    x: torch.Tensor,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    keepdims: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.linalg.matrix_norm(x, ord=ord, dim=[-2, -1], keepdim=keepdims, out=out)


matrix_norm.unsupported_dtypes = ("float16",)


matrix_norm.support_native_out = True


def matrix_power(
    x: torch.Tensor, n: int, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.linalg.matrix_power(x, n, out=out)


matrix_power.unsupported_dtypes = ("float16",)
matrix_power.support_native_out = True


# noinspection PyPep8Naming
def matrix_rank(
    x: torch.Tensor,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # ToDo: add support for default rtol value here, for the case where None is provided
    if rtol is None and out is None:
        ret = torch.linalg.matrix_rank(x)
    elif rtol is None and out is not None:
        ret = torch.linalg.matrix_rank(x, out=out)
    elif rtol is not None and out is None:
        ret = torch.linalg.matrix_rank(x,rtol)
    else:
        ret = torch.linalg.matrix_rank(x,rtol, out=out)

    ret = torch.tensor(ret, dtype=ivy.default_int_dtype(as_native=True))
    return ret


matrix_rank.support_native_out = True


def matrix_transpose(x: torch.Tensor) -> torch.Tensor:
    ret = torch.swapaxes(x, -1, -2)
    return ret


matrix_transpose.unsupported_dtypes = (
    "float16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)


def outer(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.outer(x1, x2, out=out)


outer.support_native_out = True


def pinv(
    x: torch.Tensor,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if rtol is None:
        return torch.linalg.pinv(x, out=out)
    return torch.linalg.pinv(x, rtol, out=out)


pinv.unsupported_dtypes = ("float16",)


pinv.support_native_out = True


def qr(
    x: torch.Tensor, mode: str = "reduced", out: Optional[torch.Tensor] = None
) -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    if mode == "reduced":
        q, r = torch.qr(x, some=True, out=out)
        ret = res(q, r)
    elif mode == "complete":
        q, r = torch.qr(x, some=False, out=out)
        ret = res(q, r)
    else:
        raise Exception(
            "Only 'reduced' and 'complete' qr modes are allowed for the torch backend."
        )
    return ret


qr.unsupported_dtypes = ("float16",)


def slogdet(
    x: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = torch.linalg.slogdet(x, out=out)
    return results(sign, logabsdet)


slogdet.unsupported_dtypes = ("float16",)

slogdet.support_native_out = True


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


solve.unsupported_dtypes = ("float16",)


def svd(
    x: torch.Tensor, full_matrices: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    results = namedtuple("svd", "U S Vh")

    U, D, VT = torch.linalg.svd(x, full_matrices=full_matrices)
    ret = results(U, D, VT)
    return ret


svd.unsupported_dtypes = ("float16",)


def svdvals(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.linalg.svdvals(x, out=out)


svdvals.unsupported_dtypes = ("float16",)

svdvals.support_native_out = True


def tensordot(
    x1: torch.Tensor,
    x2: torch.Tensor,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # find the type to promote to
    dtype = torch.promote_types(x1.dtype, x2.dtype)
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
    x: torch.Tensor, offset: int = 0, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    desired_dtype = x.dtype
    ret = torch.diagonal(x, offset=offset, dim1=-2, dim2=-1)
    ret = torch.sum(ret, dim=-1)
    ret = ret.type(desired_dtype)
    return ret


trace.unsupported_dtypes = ("float16",)


def vecdot(
    x1: torch.Tensor,
    x2: torch.Tensor,
    axis: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = torch.promote_types(x1.dtype, x2.dtype)
    x1, x2 = x1.type(torch.float32), x2.type(torch.float32)
    return torch.tensordot(x1, x2, dims=([axis], [axis]), out=out).type(dtype)


vecdot.unsupported_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "float16",
    "float64",
)


vecdot.support_native_out = True


def vector_norm(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
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
# ------#


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
vector_to_skew_symmetric_matrix.unsupported_dtypes = ("float16", "int8")
