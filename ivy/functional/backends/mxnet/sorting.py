from typing import Union, Optional, Literal, List
import mxnet as mx

import ivy
from ivy.utils.exceptions import IvyNotImplementedException


def argsort(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: int = (-1),
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def sort(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: int = (-1),
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def msort(
    a: Union[(None, mx.ndarray.NDArray, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def searchsorted(
    x: Union[(None, mx.ndarray.NDArray)],
    v: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    side: Literal[("left", "right")] = "left",
    sorter: Optional[Union[(ivy.Array, ivy.NativeArray, List[int])]] = None,
    ret_dtype: None = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()
