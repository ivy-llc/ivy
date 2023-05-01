import mxnet as mx
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence
from ivy import inf


def cholesky(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    upper: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.cholesky Not Implemented")


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
    raise NotImplementedError("mxnet.cross Not Implemented")


def det(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.det Not Implemented")


def diagonal(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    offset: int = 0,
    axis1: int = (-2),
    axis2: int = (-1),
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.diagonal Not Implemented")


def eig(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Tuple[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.eig Not Implemented")


def eigh(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    UPLO: str = "L",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Tuple[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.eigh Not Implemented")


def eigvalsh(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    UPLO: str = "L",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.eigvalsh Not Implemented")


def inner(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.inner Not Implemented")


def inv(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    adjoint: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.inv Not Implemented")


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
    raise NotImplementedError("mxnet.matmul Not Implemented")


def matrix_norm(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    ord: Union[(int, float, Literal[(inf, (-inf), "fro", "nuc")])] = "fro",
    axis: Tuple[(int, int)] = ((-2), (-1)),
    keepdims: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.matrix_norm Not Implemented")


def matrix_power(
    x: Union[(None, mx.ndarray.NDArray)],
    n: int,
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.matrix_power Not Implemented")


def matrix_rank(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    atol: Optional[Union[(float, Tuple[float])]] = None,
    rtol: Optional[Union[(float, Tuple[float])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.matrix_rank Not Implemented")


def matrix_transpose(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    conjugate: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.matrix_transpose Not Implemented")


def outer(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.outer Not Implemented")


def pinv(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    rtol: Optional[Union[(float, Tuple[float])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.pinv Not Implemented")


def qr(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    mode: str = "reduced",
    out: Optional[
        Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])]
    ] = None,
) -> NamedTuple:
    raise NotImplementedError("mxnet.qr Not Implemented")


def slogdet(
    x: Union[(None, mx.ndarray.NDArray)], /
) -> Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])]:
    raise NotImplementedError("mxnet.slogdet Not Implemented")


def solve(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    adjoint: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.solve Not Implemented")


def svd(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> Union[
    (Union[(None, mx.ndarray.NDArray)], Tuple[(Union[(None, mx.ndarray.NDArray)], ...)])
]:
    raise NotImplementedError("mxnet.svd Not Implemented")


def svdvals(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.svdvals Not Implemented")


def tensordot(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axes: Union[(int, Tuple[(List[int], List[int])])] = 2,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.tensordot Not Implemented")


def trace(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.trace Not Implemented")


def vecdot(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: int = (-1),
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.vecdot Not Implemented")


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
    raise NotImplementedError("mxnet.vector_norm Not Implemented")


def diag(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    k: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.diag Not Implemented")


def vander(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.vander Not Implemented")


def vector_to_skew_symmetric_matrix(
    vector: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.vector_to_skew_symmetric_matrix Not Implemented")
