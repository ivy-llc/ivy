"""
MXNet activation functions.

Collection of MXNet activation functions, wrapped to fit Ivy syntax and
signature.
"""
import mxnet as mx
import numpy as np

from ivy.utils.exceptions import IvyNotImplementedException
from typing import Optional, Union


def gelu(x: None, /, *, approximate: bool = False, out: Optional[None] = None) -> None:
    if approximate:
        return (
            0.5 * x * (1 + mx.nd.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x**3)))
        )
    return mx.nd.LeakyReLU(x, act_type="gelu")


def leaky_relu(x: None, /, *, alpha: float = 0.2, out: Optional[None] = None) -> None:
    return mx.nd.LeakyReLU(x, slope=alpha)


def relu(x: None, /, *, out: Optional[None] = None) -> None:
    return mx.nd.relu(x)


def sigmoid(x: None, /, *, out: Optional[None] = None) -> None:
    return mx.nd.sigmoid(x)


def softmax(x: None, /, *, axis: int = -1, out: Optional[None] = None) -> None:
    return mx.nd.softmax(x, axis=axis)


def softplus(
    x: Union[(int, float, mx.nd.NDArray)],
    /,
    *,
    beta: Union[(int, float)] = 1,
    threshold: Optional[Union[(int, float)]] = None,
    out: Optional[None] = None,
) -> None:
    x_beta = x * beta
    res = (
        mx.nd.add(
            mx.nd.log1p(mx.nd.exp(-mx.nd.abs(x_beta))),
            mx.nd.maximum(x_beta, 0),
        )
    ) / beta

    if threshold is not None:
        return mx.nd.where(x_beta > threshold, x, res).astype(x.dtype)
    return res.astype(x.dtype)


def log_softmax(x: None, /, *, axis: int = -1, out: Optional[None] = None):
    raise IvyNotImplementedException()


def mish(x: None, /, *, out: Optional[None] = None) -> None:
    raise IvyNotImplementedException()
