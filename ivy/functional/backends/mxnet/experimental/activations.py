from typing import Optional, Union
import mxnet as mx
from mxnet.ndarray import NDArray

from ivy.utils.exceptions import IvyNotImplementedException


def elu(
    x: NDArray,
    /,
    *,
    alpha: float = 1.0,
    out: Optional[None] = None,
) -> NDArray:
    raise NotImplementedError()


def logit(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    eps: Optional[float] = None,
    out: Optional[None] = None,
) -> None:
    raise IvyNotImplementedException()


def thresholded_relu(
    x: None, /, *, threshold: Union[(int, float)] = 0, out: Optional[None] = None
) -> None:
    raise IvyNotImplementedException()


def relu6(x: None, /, *, out: Optional[None] = None) -> None:
    raise IvyNotImplementedException()


def logsigmoid(input: None) -> None:
    raise IvyNotImplementedException()


def selu(x: None, /, *, out: Optional[None] = None) -> None:
    raise IvyNotImplementedException()


def silu(x: None, /, *, out: Optional[None] = None) -> None:
    raise IvyNotImplementedException()
