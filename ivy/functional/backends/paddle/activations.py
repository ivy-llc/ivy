"""Collection of Paddle activation functions, wrapped to fit Ivy syntax and
signature.
"""
from typing import Optional, Union

# global
import paddle as paddle

# local
import ivy
from ivy.exceptions import IvyNotImplementedException
from . import backend_version


def relu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    raise IvyNotImplementedException()


def leaky_relu(
    x: paddle.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def gelu(
    x: paddle.Tensor,
    /,
    *,
    approximate: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def sigmoid(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    raise IvyNotImplementedException()


def softmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def softplus(
    x: paddle.Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def log_softmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def mish(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    raise IvyNotImplementedException()
