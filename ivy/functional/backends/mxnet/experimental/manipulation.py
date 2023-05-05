from typing import Union, Optional, Sequence, Tuple, List
from numbers import Number
import mxnet as mx


def moveaxis(
    a: Union[(None, mx.ndarray.NDArray)],
    source: Union[(int, Sequence[int])],
    destination: Union[(int, Sequence[int])],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.moveaxis Not Implemented")


def heaviside(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.heaviside Not Implemented")


def flipud(
    m: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.flipud Not Implemented")


def vstack(
    arrays: Union[(Sequence[None], Sequence[mx.ndarray.NDArray])],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.vstack Not Implemented")


def hstack(
    arrays: Union[(Sequence[None], Sequence[mx.ndarray.NDArray])],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.hstack Not Implemented")


def rot90(
    m: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    k: int = 1,
    axes: Tuple[(int, int)] = (0, 1),
    out: Union[(None, mx.ndarray.NDArray)] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.rot90 Not Implemented")


def top_k(
    x: None,
    k: int,
    /,
    *,
    axis: int = (-1),
    largest: bool = True,
    out: Optional[Tuple[(None, None)]] = None,
) -> Tuple[(None, None)]:
    raise NotImplementedError("mxnet.top_k Not Implemented")


def fliplr(
    m: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.fliplr Not Implemented")


def i0(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.i0 Not Implemented")


def vsplit(
    ary: Union[(None, mx.ndarray.NDArray)],
    indices_or_sections: Union[(int, Tuple[(int, ...)])],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.vsplit Not Implemented")


def dsplit(
    ary: Union[(None, mx.ndarray.NDArray)],
    indices_or_sections: Union[(int, Tuple[(int, ...)])],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.dsplit Not Implemented")


def atleast_1d(
    *arys: Union[(None, mx.ndarray.NDArray, bool, Number)], copy: Optional[bool] = None
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.atleast_1d Not Implemented")


def dstack(
    arrays: Union[(Sequence[None], Sequence[mx.ndarray.NDArray])],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.dstack Not Implemented")


def atleast_2d(
    *arys: Union[(None, mx.ndarray.NDArray)], copy: Optional[bool] = None
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.atleast_2d Not Implemented")


def atleast_3d(
    *arys: Union[(None, mx.ndarray.NDArray, bool, Number)], copy: Optional[bool] = None
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.atleast_3d Not Implemented")


def take_along_axis(
    arr: Union[(None, mx.ndarray.NDArray)],
    indices: Union[(None, mx.ndarray.NDArray)],
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.take_along_axis Not Implemented")


def hsplit(
    ary: Union[(None, mx.ndarray.NDArray)],
    indices_or_sections: Union[(int, Tuple[(int, ...)])],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.hsplit Not Implemented")


def broadcast_shapes(*shapes: Union[(List[int], List[Tuple])]) -> Tuple[(int, ...)]:
    raise NotImplementedError("mxnet.broadcast_shapes Not Implemented")


def expand(
    x: Union[(None, mx.ndarray.NDArray)],
    shape: Union[(List[int], List[Tuple])],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.expand Not Implemented")


def concat_from_sequence(
    input_sequence: Union[(Tuple[None], List[None])],
    /,
    *,
    new_axis: int = 0,
    axis: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.concat_from_sequence Not Implemented")
