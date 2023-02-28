"""Collection of Mindspore activation functions, wrapped to fit Ivy syntax and
signature.
"""
from typing import Optional, Union

# global
import mindspore as ms
from mindspore import ops
import mindspore.numpy as np

# local
import ivy


def relu(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return ops.relu(x)


def leaky_relu(
    x: ms.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return ops.leaky_relu(x, alpha)

def gelu(
    x: ms.Tensor,
    /,
    *,
    approximate: bool = False,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if approximate:
        return (
            0.5 * x * (1 + ops.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x**3)))
        )
    return ops.gelu(x)


def sigmoid(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    if not ivy.is_array(x):
        x = ms.Tensor(x)
    return ops.sigmoid(x)

sigmoid.support_native_out = True


def softmax(
    x: ms.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if axis is None:
        axis = -1
    return ops.softmax(x, axis)


def softplus(
    x: ms.Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = 1,
    threshold: Optional[Union[int, float]] = 20,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return ops.log(1 + ops.exp(beta * x)) / beta


def log_softmax(
    x: ms.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[ms.Tensor] = None,
):
    return ops.log_softmax(x, axis)


def mish(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return x * ops.tanh(softplus(x))
