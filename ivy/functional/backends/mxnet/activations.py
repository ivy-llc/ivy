"""Collection of MXNet activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional

# global
import numpy as np
import mxnet as mx

# local
import ivy


def relu(x: mx.nd.NDArray, out: Optional[mx.nd.NDArray] = None) -> mx.nd.NDArray:
    ret = mx.nd.relu(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def leaky_relu(x: mx.nd.NDArray, alpha: Optional[float] = 0.2) -> mx.nd.NDArray:
    return mx.nd.LeakyReLU(x, slope=alpha)


def gelu(x, approximate=True):
    if approximate:
        return (
            0.5 * x * (1 + mx.nd.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x**3)))
        )
    return mx.nd.LeakyReLU(x, act_type="gelu")


def sigmoid(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.sigmoid(x)


def tanh(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.tanh(x)


def softmax(x: mx.nd.NDArray, axis: Optional[int] = -1) -> mx.nd.NDArray:
    return mx.nd.softmax(x, axis=axis)


def softplus(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return mx.nd.log(mx.nd.exp(x) + 1)
