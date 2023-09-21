from typing import Tuple, Union, Optional
import mxnet as mx

from ivy.utils.exceptions import IvyNotImplementedException


def unique_all(
    x: Union[(None, mx.ndarray.NDArray)], /, *, axis: Optional[int] = None
) -> Tuple[
    (
        Union[(None, mx.ndarray.NDArray)],
        Union[(None, mx.ndarray.NDArray)],
        Union[(None, mx.ndarray.NDArray)],
        Union[(None, mx.ndarray.NDArray)],
    )
]:
    raise IvyNotImplementedException()


def unique_counts(
    x: Union[(None, mx.ndarray.NDArray)], /
) -> Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])]:
    raise IvyNotImplementedException()


def unique_inverse(
    x: Union[(None, mx.ndarray.NDArray)], /
) -> Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])]:
    raise IvyNotImplementedException()


def unique_values(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()
