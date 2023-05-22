from typing import Optional, Union
import mxnet as mx
from mxnet.ndarray import NDArray

from ivy.utils.exceptions import IvyNotImplementedException

def elu(
    x: NDArray,
    alpha: float = 1.0,
    out: Optional[NDArray] = None,
    inplace: bool = False,
) -> NDArray:
    if inplace and out is not None:
        raise ValueError("Cannot specify both 'inplace' and 'out' parameters.")
    if inplace:
        x[:] = mx.nd.where(x > 0, x, alpha * (mx.nd.exp(x) - 1))
        return x
    else:
        result = mx.nd.where(x > 0, x, alpha * (mx.nd.exp(x) - 1))
        if out is not None:
            out[:] = result
            return out
        else:
            return result

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
