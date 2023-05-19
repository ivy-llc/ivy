"""
MXNet activation functions.

Collection of MXNet activation functions, wrapped to fit Ivy syntax and
signature.
"""
from ivy.utils.exceptions import IvyNotImplementedException
from typing import Optional, Union
import ivy


def gelu(x: None, /, *, approximate: bool = False, out: Optional[None] = None) -> None:
    raise IvyNotImplementedException()


def leaky_relu(x: None, /, *, alpha: float = 0.2, out: Optional[None] = None) -> None:
    raise IvyNotImplementedException()


def relu(x: None, /, *, out: Optional[None] = None) -> None:
    raise IvyNotImplementedException()


def sigmoid(x: None, /, *, out: Optional[None] = None) -> None:
    raise IvyNotImplementedException()


def softmax(
    x: None, /, *, axis: Optional[int] = None, out: Optional[None] = None
) -> None:
    raise IvyNotImplementedException()


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
