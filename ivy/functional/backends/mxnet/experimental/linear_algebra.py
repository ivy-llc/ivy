from typing import Union, Optional, Tuple, List, Sequence
import mxnet as mx


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
    raise NotImplementedError("mxnet.eigh_tridiagonal Not Implemented")


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
    raise NotImplementedError("mxnet.diagflat Not Implemented")


def kron(
    a: Union[(None, mx.ndarray.NDArray)],
    b: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.kron Not Implemented")


def matrix_exp(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.matrix_exp Not Implemented")


def eig(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Tuple[None]:
    raise NotImplementedError("mxnet.eig Not Implemented")


def eigvals(
    x: Union[(None, mx.ndarray.NDArray)], /
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.eigvals Not Implemented")


def adjoint(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.adjoint Not Implemented")


def multi_dot(
    x: Sequence[Union[(None, mx.ndarray.NDArray)]],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> None:
    raise NotImplementedError("mxnet.multi_dot Not Implemented")


def cond(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    p: Optional[Union[(None, int, str)]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.cond Not Implemented")


def cov(
    x1: None,
    x2: None = None,
    /,
    *,
    rowVar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[None] = None,
    aweights: Optional[None] = None,
    dtype: Optional[type] = None,
) -> None:
    raise NotImplementedError("mxnet.cov Not Implemented")
