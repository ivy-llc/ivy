"""
Collection of PyTorch math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch


def sin(x):
    return _torch.sin(x)


def cos(x):
    return _torch.cos(x)


def tan(x):
    return _torch.tan(x)


def asin(x):
    return _torch.asin(x)


def acos(x):
    return _torch.acos(x)


def atan(x):
    return _torch.atan(x)


def atan2(x, y):
    return _torch.atan2(x, y)


def sinh(x):
    return _torch.sinh(x)


def cosh(x):
    return _torch.cosh(x)


def tanh(x):
    return _torch.tanh(x)


def asinh(x):
    return _torch.asinh(x)


def acosh(x):
    return _torch.acosh(x)


def atanh(x):
    return _torch.atanh(x)


def log(x):
    return _torch.log(x)


def exp(x):
    return _torch.exp(x)


def erf(x):
    return _torch.erf(x)
