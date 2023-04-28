from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence
from ivy import inf


def cholesky(
    x: Union[(None, tf.Variable)],
    /,
    *,
    upper: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.cholesky Not Implemented")


def cross(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    axisa: int = (-1),
    axisb: int = (-1),
    axisc: int = (-1),
    axis: Optional[int] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.cross Not Implemented")


def det(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.det Not Implemented")


def diagonal(
    x: Union[(None, tf.Variable)],
    /,
    *,
    offset: int = 0,
    axis1: int = (-2),
    axis2: int = (-1),
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.diagonal Not Implemented")


def eig(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Tuple[Union[(None, tf.Variable)]]:
    raise NotImplementedError("mxnet.eig Not Implemented")


def eigh(
    x: Union[(None, tf.Variable)],
    /,
    *,
    UPLO: str = "L",
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Tuple[Union[(None, tf.Variable)]]:
    raise NotImplementedError("mxnet.eigh Not Implemented")


def eigvalsh(
    x: Union[(None, tf.Variable)],
    /,
    *,
    UPLO: str = "L",
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.eigvalsh Not Implemented")


def inner(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.inner Not Implemented")


def inv(
    x: Union[(None, tf.Variable)],
    /,
    *,
    adjoint: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.inv Not Implemented")


def matmul(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.matmul Not Implemented")


def matrix_norm(
    x: Union[(None, tf.Variable)],
    /,
    *,
    ord: Union[(int, float, Literal[(inf, (-inf), "fro", "nuc")])] = "fro",
    axis: Tuple[(int, int)] = ((-2), (-1)),
    keepdims: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.matrix_norm Not Implemented")


def matrix_power(
    x: Union[(None, tf.Variable)],
    n: int,
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.matrix_power Not Implemented")


def matrix_rank(
    x: Union[(None, tf.Variable)],
    /,
    *,
    atol: Optional[Union[(float, Tuple[float])]] = None,
    rtol: Optional[Union[(float, Tuple[float])]] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.matrix_rank Not Implemented")


def matrix_transpose(
    x: Union[(None, tf.Variable)],
    /,
    *,
    conjugate: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.matrix_transpose Not Implemented")


def outer(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.outer Not Implemented")


def pinv(
    x: Union[(None, tf.Variable)],
    /,
    *,
    rtol: Optional[Union[(float, Tuple[float])]] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.pinv Not Implemented")


def qr(
    x: Union[(None, tf.Variable)],
    /,
    *,
    mode: str = "reduced",
    out: Optional[
        Tuple[(Union[(None, tf.Variable)], Union[(None, tf.Variable)])]
    ] = None,
) -> NamedTuple:
    raise NotImplementedError("mxnet.qr Not Implemented")


def slogdet(
    x: Union[(None, tf.Variable)], /
) -> Tuple[(Union[(None, tf.Variable)], Union[(None, tf.Variable)])]:
    raise NotImplementedError("mxnet.slogdet Not Implemented")


def solve(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    adjoint: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.solve Not Implemented")


def svd(
    x: Union[(None, tf.Variable)],
    /,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> Union[(Union[(None, tf.Variable)], Tuple[(Union[(None, tf.Variable)], ...)])]:
    raise NotImplementedError("mxnet.svd Not Implemented")


def svdvals(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.svdvals Not Implemented")


def tensordot(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    axes: Union[(int, Tuple[(List[int], List[int])])] = 2,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.tensordot Not Implemented")


def trace(
    x: Union[(None, tf.Variable)],
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.trace Not Implemented")


def vecdot(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    axis: int = (-1),
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.vecdot Not Implemented")


def vector_norm(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    keepdims: bool = False,
    ord: Union[(int, float, Literal[(inf, (-inf))])] = 2,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.vector_norm Not Implemented")


def diag(
    x: Union[(None, tf.Variable)],
    /,
    *,
    k: int = 0,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.diag Not Implemented")


def vander(
    x: Union[(None, tf.Variable)],
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.vander Not Implemented")


def vector_to_skew_symmetric_matrix(
    vector: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.vector_to_skew_symmetric_matrix Not Implemented")
