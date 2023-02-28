# global

import paddle
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence

from collections import namedtuple


# local
import ivy
from ivy import inf
from ivy.utils.exceptions import IvyNotImplementedException
from . import backend_version


# Array API Standard #
# -------------------#


def cholesky(
    x: paddle.Tensor, /, *, upper: bool = False, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    paddle.linalg.cholesky(x, upper=upper)


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

    x1, x2 = ivy.promote_types_of_inputs(x1, x2)

    if axis is not None:
        return paddle.cross(x1, x2, axis=axis)
    x1 = paddle.moveaxis(x1, axisa, 1)
    x2 = paddle.moveaxis(x2, axisb, 1)
    ret = paddle.cross(x1, x2)
    return paddle.moveaxis(ret, 1, axisc)


def det(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.linalg.det(x)


def diagonal(
    x: paddle.Tensor,
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
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
    return paddle.inner(x1, x2)


def inv(
    x: paddle.Tensor,
    /,
    *,
    adjoint: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


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

    if adjoint_a:
        x1 = paddle.moveaxis(x1, -2, -1).conj()
    if adjoint_b:
        x2 = paddle.moveaxis(x2, -2, -1).conj()
    return paddle.matmul(x1, x2, transpose_x=transpose_a, transpose_y=transpose_b)


def matrix_norm(
    x: paddle.Tensor,
    /,
    *,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    axis: Optional[Tuple[int, int]] = (-2, -1),
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


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
    raise IvyNotImplementedException()


def matrix_rank(
    x: paddle.Tensor,
    /,
    *,
    atol: Optional[Union[float, Tuple[float]]] = None,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
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
            return paddle.to_tensor(0).astype(x.dtype)
    elif len(x.shape) > 3:
        if x.shape[-3] == 0 or x.shape[-4] == 0:
            return paddle.to_tensor(0).astype(x.dtype)
    axis = None
    ret_shape = x.shape[:-2]
    if len(x.shape) == 2:
        singular_values = paddle.linalg.svd(x, full_matrices=False)
    elif len(x.shape) > 2:
        y = x.reshape((-1, *x.shape[-2:]))
        singular_values = paddle.to_tensor(
            [
                paddle.linalg.svd(split[0], full_matrices=False)[1]
                for split in paddle.split(y, y.shape[0], axis=0)
            ]
        )
        axis = 1
    if len(x.shape) < 2 or len(singular_values.shape) == 0:
        return paddle.to_tensor(0).astype(x.dtype)
    max_values = paddle.max(singular_values, axis=axis)
    if atol is None:
        if rtol is None:
            ret = paddle.sum(singular_values != 0, axis=axis)
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
            ret = paddle.sum(singular_values > atol, axis=axis)
        else:
            tol = paddle.max(atol, max_values * rtol)
            ret = paddle.sum(singular_values > tol, axis=axis)
    if len(ret_shape):
        ret = ret.reshape(ret_shape)
    return ret.astype(x.dtype)


def matrix_transpose(
    x: paddle.Tensor, /, *, conjugate: bool = False, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def outer(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.outer(x1, x2)


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
    raise IvyNotImplementedException()


def qr(
    x: paddle.Tensor,
    /,
    *,
    mode: str = "reduced",
    out: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def slogdet(
    x: paddle.Tensor,
    /,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def solve(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    adjoint: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def svd(
    x: paddle.Tensor, /, *, full_matrices: bool = True, compute_uv: bool = True
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
    raise IvyNotImplementedException()


def svdvals(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def tensordot(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    axes: Union[int, Tuple[List[int], List[int]]] = 2,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def trace(
    x: paddle.Tensor,
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def vecdot(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    axis: int = -1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


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
    raise IvyNotImplementedException()


# Extra #
# ----- #


def diag(
    x: paddle.Tensor,
    /,
    *,
    k: int = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.diag(x, k=k)


def vander(
    x: paddle.Tensor,
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def vector_to_skew_symmetric_matrix(
    vector: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
