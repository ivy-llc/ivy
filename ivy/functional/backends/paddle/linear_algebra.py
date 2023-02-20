# global

import paddle
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence

from collections import namedtuple


# local
import ivy
from ivy import inf
from ivy.exceptions import IvyNotImplementedException
from . import backend_version


# Array API Standard #
# -------------------#


def cholesky(
    x: paddle.Tensor, /, *, upper: bool = False, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


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

    raise IvyNotImplementedException()


def det(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    raise IvyNotImplementedException()


def diagonal(
    x: paddle.Tensor,
    /,
    *,
    offset: int = 0,
    axis1: int = -2,
    axis2: int = -1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def eigh(
    x: paddle.Tensor, /, *, UPLO: Optional[str] = "L", out: Optional[paddle.Tensor] = None
) -> Tuple[paddle.Tensor]:
    raise IvyNotImplementedException()


def eigvalsh(
    x: paddle.Tensor, /, *, UPLO: Optional[str] = "L", out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def inner(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


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
    raise IvyNotImplementedException()


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
    raise IvyNotImplementedException()


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
    raise IvyNotImplementedException()


def matrix_transpose(
    x: paddle.Tensor, /, *, conjugate: bool = False, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def outer(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def pinv(
    x: paddle.Tensor,
    /,
    *,
    rtol: Optional[Union[float, Tuple[float]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


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


def svdvals(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
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
    raise IvyNotImplementedException()


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
