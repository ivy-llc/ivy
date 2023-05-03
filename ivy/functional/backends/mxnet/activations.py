"""
MXNet activation functions.

Collection of MXNet activation functions, wrapped to fit Ivy syntax and
signature.
"""
from typing import Optional, Union
import ivy


def gelu(x: None, /, *, approximate: bool = False, out: Optional[None] = None) -> None:
    raise NotImplementedError("mxnet.gelu Not Implemented")


def leaky_relu(x: None, /, *, alpha: float = 0.2, out: Optional[None] = None) -> None:
    raise NotImplementedError("mxnet.leaky_relu Not Implemented")


def relu(x: None, /, *, out: Optional[None] = None) -> None:
    raise NotImplementedError("mxnet.relu Not Implemented")


def sigmoid(x: None, /, *, out: Optional[None] = None) -> None:
    raise NotImplementedError("mxnet.sigmoid Not Implemented")


def softmax(
    x: None, /, *, axis: Optional[int] = None, out: Optional[None] = None
) -> None:
    raise NotImplementedError("mxnet.softmax Not Implemented")


def softplus(
    x: None,
    /,
    *,
    beta: Optional[Union[(int, float)]] = None,
    threshold: Optional[Union[(int, float)]] = None,
    out: Optional[None] = None,
) -> None:
    raise NotImplementedError("mxnet.softplus Not Implemented")


def log_softmax(x: None, /, *, axis: Optional[int] = None, out: Optional[None] = None):
    raise NotImplementedError("mxnet.log_softmax Not Implemented")


def deserialize(
    name: Union[(str, None)], /, *, custom_objects: Optional[ivy.Dict] = None
) -> Union[(ivy.Callable, None)]:
    raise NotImplementedError("mxnet.deserialize Not Implemented")


def get(
    identifier: Union[(str, ivy.Callable, None)],
    /,
    *,
    custom_objects: Optional[ivy.Dict] = None,
) -> Union[(ivy.Callable, None)]:
    raise NotImplementedError("mxnet.get Not Implemented")


def mish(x: None, /, *, out: Optional[None] = None) -> None:
    raise NotImplementedError("mxnet.mish Not Implemented")
