from typing import Union, Optional, Tuple, List, Sequence


def eigh_tridiagonal(
    alpha: Union[(None, tf.Variable)],
    beta: Union[(None, tf.Variable)],
    /,
    *,
    eigvals_only: bool = True,
    select: str = "a",
    select_range: Optional[
        Union[(Tuple[(int, int)], List[int], None, tf.Variable)]
    ] = None,
    tol: Optional[float] = None,
) -> Union[
    (None, tf.Variable, Tuple[(Union[(None, tf.Variable)], Union[(None, tf.Variable)])])
]:
    raise NotImplementedError("mxnet.eigh_tridiagonal Not Implemented")


def diagflat(
    x: Union[(None, tf.Variable)],
    /,
    *,
    offset: int = 0,
    padding_value: float = 0,
    align: str = "RIGHT_LEFT",
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
):
    raise NotImplementedError("mxnet.diagflat Not Implemented")


def kron(
    a: Union[(None, tf.Variable)],
    b: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.kron Not Implemented")


def matrix_exp(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.matrix_exp Not Implemented")


def eig(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Tuple[None]:
    raise NotImplementedError("mxnet.eig Not Implemented")


def eigvals(x: Union[(None, tf.Variable)], /) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.eigvals Not Implemented")


def adjoint(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.adjoint Not Implemented")


def multi_dot(
    x: Sequence[Union[(None, tf.Variable)]],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> None:
    raise NotImplementedError("mxnet.multi_dot Not Implemented")


def cond(
    x: Union[(None, tf.Variable)],
    /,
    *,
    p: Optional[Union[(None, int, str)]] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
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
