from typing import Union, Optional, Sequence, Tuple, List
from numbers import Number
import mxnet as mx

from ivy.utils.exceptions import IvyNotImplementedException


def moveaxis(
    a: Union[(None, mx.ndarray.NDArray)],
    source: Union[(int, Sequence[int])],
    destination: Union[(int, Sequence[int])],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def heaviside(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def flipud(
    m: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def vstack(
    arrays: Union[(Sequence[None], Sequence[mx.ndarray.NDArray])],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def hstack(
    arrays: Union[(Sequence[None], Sequence[mx.ndarray.NDArray])],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def rot90(
    m: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    k: int = 1,
    axes: Tuple[(int, int)] = (0, 1),
    out: Union[(None, mx.ndarray.NDArray)] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def top_k(
    x: None,
    k: int,
    /,
    *,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
    out: Optional[Tuple[(None, None)]] = None,
) -> Tuple[(None, None)]:
    raise IvyNotImplementedException()


def fliplr(
    m: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def i0(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def vsplit(
    ary: Union[(None, mx.ndarray.NDArray)],
    indices_or_sections: Union[(int, Tuple[(int, ...)])],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()


def dsplit(
    ary: Union[(None, mx.ndarray.NDArray)],
    indices_or_sections: Union[(int, Tuple[(int, ...)])],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()


def atleast_1d(
    *arys: Union[(None, mx.ndarray.NDArray, bool, Number)], copy: Optional[bool] = None
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()


def dstack(
    arrays: Union[(Sequence[None], Sequence[mx.ndarray.NDArray])],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def atleast_2d(
    *arys: Union[(None, mx.ndarray.NDArray)], copy: Optional[bool] = None
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()


def atleast_3d(
    *arys: Union[(None, mx.ndarray.NDArray, bool, Number)], copy: Optional[bool] = None
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()


def take_along_axis(
    arr: Union[(None, mx.ndarray.NDArray)],
    indices: Union[(None, mx.ndarray.NDArray)],
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def hsplit(
    ary: Union[(None, mx.ndarray.NDArray)],
    indices_or_sections: Union[(int, Tuple[(int, ...)])],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise IvyNotImplementedException()


def broadcast_shapes(*shapes: Union[(List[int], List[Tuple])]) -> Tuple[(int, ...)]:
    raise IvyNotImplementedException()


def expand(
    x: Union[(None, mx.ndarray.NDArray)],
    shape: Union[(List[int], List[Tuple])],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def concat_from_sequence(
    input_sequence: Union[(Tuple[None], List[None])],
    /,
    *,
    new_axis: int = 0,
    axis: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()
