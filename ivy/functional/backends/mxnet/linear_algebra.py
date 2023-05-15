import mxnet as mx
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence

from ivy import inf
from ivy.utils.exceptions import IvyNotImplementedException


def cholesky(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    upper: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def cross(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axisa: int = (-1),
    axisb: int = (-1),
    axisc: int = (-1),
    axis: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def det(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def diagonal(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    offset: int = 0,
    axis1: int = (-2),
    axis2: int = (-1),
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def eig(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Tuple[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()


def eigh(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    UPLO: str = "L",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Tuple[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()


def eigvalsh(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    UPLO: str = "L",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def inner(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def inv(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    adjoint: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def matmul(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def matrix_norm(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    ord: Union[(int, float, Literal[(inf, (-inf), "fro", "nuc")])] = "fro",
    axis: Tuple[(int, int)] = ((-2), (-1)),
    keepdims: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def matrix_power(
    x: Union[(None, mx.ndarray.NDArray)],
    n: int,
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def matrix_rank(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    atol: Optional[Union[(float, Tuple[float])]] = None,
    rtol: Optional[Union[(float, Tuple[float])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def matrix_transpose(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    conjugate: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def outer(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def pinv(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    rtol: Optional[Union[(float, Tuple[float])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def qr(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    mode: str = "reduced",
    out: Optional[
        Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])]
    ] = None,
) -> NamedTuple:
    raise IvyNotImplementedException()


def slogdet(
    x: Union[(None, mx.ndarray.NDArray)], /
) -> Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])]:
    raise IvyNotImplementedException()


def solve(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    adjoint: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def svd(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> Union[
    (Union[(None, mx.ndarray.NDArray)], Tuple[(Union[(None, mx.ndarray.NDArray)], ...)])
]:
    raise IvyNotImplementedException()


def svdvals(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def tensordot(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axes: Union[(int, Tuple[(List[int], List[int])])] = 2,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def trace(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def vecdot(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: int = (-1),
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def vector_norm(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    keepdims: bool = False,
    ord: Union[(int, float, Literal[(inf, (-inf))])] = 2,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def diag(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    k: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def vander(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def vector_to_skew_symmetric_matrix(
    vector: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()
