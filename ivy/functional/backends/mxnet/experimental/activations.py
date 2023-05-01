from typing import Optional, Union
import mxnet as mx


def logit(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    eps: Optional[float] = None,
    out: Optional[None] = None,
) -> None:
    raise NotImplementedError("mxnet.logit Not Implemented")


def thresholded_relu(
    x: None, /, *, threshold: Union[(int, float)] = 0, out: Optional[None] = None
) -> None:
    raise NotImplementedError("mxnet.thresholded_relu Not Implemented")


def relu6(x: None, /, *, out: Optional[None] = None) -> None:
    raise NotImplementedError("mxnet.relu6 Not Implemented")


def logsigmoid(input: None) -> None:
    raise NotImplementedError("mxnet.logsigmoid Not Implemented")


def selu(x: None, /, *, out: Optional[None] = None) -> None:
    raise NotImplementedError("mxnet.selu Not Implemented")
