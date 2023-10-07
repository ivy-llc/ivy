from typing import Optional, Union
import mxnet as mx

from ivy.utils.exceptions import IvyNotImplementedException


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


def celu(
    x: None, /, *, alpha: float = 0.2, complex_mode="jax", out: Optional[None] = None
) -> None:
    return mx.nd.maximum(0, x) + alpha * mx.nd.expm1(mx.nd.minimum(0, x) / alpha)
