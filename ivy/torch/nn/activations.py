"""
Collection of PyTorch activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np
import torch as _torch


def relu(x):
    return _torch.relu(x)


def leaky_relu(x, alpha: float = 0.2):
    return _torch.nn.functional.leaky_relu(x, alpha)


def gelu(x, approximate: bool = True):
    if approximate:
        return 0.5 * x * (1 + _torch.tanh(((2 / _np.pi) ** 0.5) * (x + 0.044715 * x ** 3)))
    return _torch.nn.functional.gelu(x)


def tanh(x):
    return _torch.tanh(x)


def sigmoid(x):
    return _torch.sigmoid(x)


def softmax(x, axis: int = -1):
    return _torch.softmax(x, axis)


def softplus(x):
    return _torch.nn.functional.softplus(x)
