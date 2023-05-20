"""
MXNet activation functions.

Collection of MXNet activation functions, wrapped to fit Ivy syntax and
signature.
"""
import mxnet as mx
import numpy as np

from ivy.utils.exceptions import IvyNotImplementedException
from typing import Optional, Union
import ivy


def gelu(x: None, /, *, approximate: bool = False, out: Optional[None] = None) -> None:
    if approximate:
        return 0.5 * x * (1 + mx.nd.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x ** 3)))
    return mx.nd.LeakyReLU(x, act_type='gelu')


def leaky_relu(x: None, /, *, alpha: float = 0.2, out: Optional[None] = None) -> None:
    mx.nd.LeakyReLU(x, slope=alpha)


def relu(x: None, /, *, out: Optional[None] = None) -> None:
    return mx.nd.relu(x)


def sigmoid(x: None, /, *, out: Optional[None] = None) -> None:
    raise mx.nd.sigmoid(x)


def softmax(
    x: None, /, *, axis: Optional[int] = None, out: Optional[None] = None
) -> None:
    raise mx.nd.softmax(x, axis=axis)


def softplus(
    x: None,
    /,
    *,
    beta: Optional[Union[(int, float)]] = None,
    threshold: Optional[Union[(int, float)]] = None,
    out: Optional[None] = None,
) -> None:
    raise IvyNotImplementedException()


def log_softmax(x: None, /, *, axis: Optional[int] = None, out: Optional[None] = None):
    raise IvyNotImplementedException()


def deserialize(
    name: Union[(str, None)], /, *, custom_objects: Optional[ivy.Dict] = None
) -> Union[(ivy.Callable, None)]:
    raise IvyNotImplementedException()


def get(
    identifier: Union[(str, ivy.Callable, None)],
    /,
    *,
    custom_objects: Optional[ivy.Dict] = None,
) -> Union[(ivy.Callable, None)]:
    raise IvyNotImplementedException()


def mish(x: None, /, *, out: Optional[None] = None) -> None:
    raise IvyNotImplementedException()
