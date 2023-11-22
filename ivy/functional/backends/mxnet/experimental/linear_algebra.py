from typing import Union, Optional, Tuple, List, Sequence
import mxnet as mx

from ivy.utils.exceptions import IvyNotImplementedException


def eigh_tridiagonal(
    alpha: Union[(None, mx.ndarray.NDArray)],
    beta: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    eigvals_only: bool = True,
    select: str = "a",
    select_range: Optional[
        Union[(Tuple[(int, int)], List[int], None, mx.ndarray.NDArray)]
    ] = None,
    tol: Optional[float] = None,
) -> Union[
    (
        None,
        mx.ndarray.NDArray,
        Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])],
    )
]:
    raise IvyNotImplementedException()


def diagflat(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    offset: int = 0,
    padding_value: float = 0,
    align: str = "RIGHT_LEFT",
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
):
    raise IvyNotImplementedException()


def kron(
    a: Union[(None, mx.ndarray.NDArray)],
    b: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def matrix_exp(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def eig(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Tuple[None]:
    raise IvyNotImplementedException()


def eigvals(
    x: Union[(None, mx.ndarray.NDArray)], /
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def adjoint(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def solve_triangular(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    upper: bool = True,
    adjoint: bool = False,
    unit_diagonal: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    # Multiplying with a mask matrix can stop gradients on the diagonal.
    if unit_diagonal:
        w = mx.eye(x1.shape[-2], batch_shape=x1.shape[:-2], dtype=x1.dtype)
        x1 = w + (1 - w) * x1
    # MXNet does not support complex tensors for this operation,
    # so adjoint always equals transpose.
    return mx.nd.linalg.trsm(x1, x2, lower=not upper, transpose=adjoint)


def multi_dot(
    x: Sequence[Union[(None, mx.ndarray.NDArray)]],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> None:
    raise IvyNotImplementedException()


def cond(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    p: Optional[Union[(None, int, str)]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def dot(
    a: mx.ndarray.NDArray,
    b: mx.ndarray.NDArray,
    /,
    *,
    out: Optional[mx.ndarray.NDArray] = None,
) -> mx.ndarray.NDArray:
    return mx.symbol.dot(a, b, out=out)


dot.support_native_out = True
