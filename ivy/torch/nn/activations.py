"""
Collection of PyTorch activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch


def relu(x):
    return _torch.nn.functional.relu(x)


def leaky_relu(x, alpha:float=0.2):
    return _torch.nn.functional.leaky_relu(x, alpha)


def tanh(x):
    return _torch.nn.functional.tanh(x)


def sigmoid(x):
    return _torch.nn.functional.sigmoid(x)


def softmax(x, axis: int = -1):
    return _torch.nn.functional.softmax(x, axis)


def softplus(x):
    return _torch.nn.functional.softplus(x)
