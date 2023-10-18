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


def take(
    x: Union[int, List, Union[(None, mx.ndarray.NDArray)]],
    indices: Union[int, List, Union[(None, mx.ndarray.NDArray)]],
    /,
    *,
    axis: Optional[int] = None,
    mode: str = "clip",
    fill_value: Optional[Number] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
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


def index_add(
    x: Union[(None, mx.ndarray.NDArray)],
    index: Union[(None, mx.ndarray.NDArray)],
    axis: int,
    value: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    name: Optional[str] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    x = mx.nd.swapaxes(x, axis, 0)
    value = mx.nd.swapaxes(value, axis, 0)
    _to_adds = []
    index = sorted(
        zip(index.asnumpy().tolist(), range(len(index))), key=(lambda i: i[0])
    )
    while index:
        _curr_idx = index[0][0]
        while len(_to_adds) < _curr_idx:
            _to_adds.append(mx.nd.zeros_like(value[0]))
        _to_add_cum = value[index[0][1]]
        while len(index) > 1 and (index[0][0] == index[1][0]):
            _to_add_cum = _to_add_cum + value[index.pop(1)[1]]
        index.pop(0)
        _to_adds.append(_to_add_cum)
    while len(_to_adds) < x.shape[0]:
        _to_adds.append(mx.nd.zeros_like(value[0]))
    _to_adds = mx.nd.stack(*_to_adds)
    if len(x.shape) < 2:
        # Added this line due to the paddle backend treating scalars as 1-d arrays
        _to_adds = mx.nd.flatten(_to_adds)

    ret = mx.nd.add(x, _to_adds)
    ret = mx.nd.swapaxes(ret, axis, 0)
    return ret
