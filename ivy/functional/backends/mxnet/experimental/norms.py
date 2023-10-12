from typing import Union, Optional, Tuple
import mxnet as mx

from ivy.utils.exceptions import IvyNotImplementedException


def l2_normalize(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[None] = None,
) -> None:
    raise IvyNotImplementedException()


def batch_norm(
    x: Union[(None, mx.ndarray.NDArray)],
    mean: Union[(None, mx.ndarray.NDArray)],
    variance: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    scale: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    offset: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    training: bool = False,
    eps: float = 1e-05,
    momentum: float = 0.1,
    out: Optional[None] = None,
) -> Tuple[
    (
        Union[(None, mx.ndarray.NDArray)],
        Union[(None, mx.ndarray.NDArray)],
        Union[(None, mx.ndarray.NDArray)],
    )
]:
    raise IvyNotImplementedException()


def instance_norm(
    x: Union[(None, mx.ndarray.NDArray)],
    mean: Union[(None, mx.ndarray.NDArray)],
    variance: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    scale: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    offset: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    training: bool = False,
    eps: float = 1e-05,
    momentum: float = 0.1,
    out: Optional[None] = None,
) -> Tuple[
    (
        Union[(None, mx.ndarray.NDArray)],
        Union[(None, mx.ndarray.NDArray)],
        Union[(None, mx.ndarray.NDArray)],
    )
]:
    raise IvyNotImplementedException()


def lp_normalize(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[None] = None,
) -> None:
    raise IvyNotImplementedException()
