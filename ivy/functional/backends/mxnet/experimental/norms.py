from typing import Union, Optional, Tuple
import mxnet as mx


def l2_normalize(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[None] = None,
) -> None:
    raise NotImplementedError("mxnet.l2_normalize Not Implemented")


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
    raise NotImplementedError("mxnet.batch_norm Not Implemented")


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
    raise NotImplementedError("mxnet.instance_norm Not Implemented")


def lp_normalize(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[None] = None,
) -> None:
    raise NotImplementedError("mxnet.lp_normalize Not Implemented")
