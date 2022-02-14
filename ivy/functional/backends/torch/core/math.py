"""
Collection of PyTorch math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import math
import torch as _torch


def sin(x):
    if isinstance(x, float):
        return math.sin(x)
    return _torch.sin(x)


def cos(x):
    if isinstance(x, float):
        return math.cos(x)
    if x.dtype == _torch.float16:
        return _torch.cos(_torch.tensor(x, dtype=_torch.float32))
    return _torch.cos(x)


def tan(x):
    if isinstance(x, float):
        return math.tan(x)
    return _torch.tan(x)


def asin(x):
    if isinstance(x, float):
        return math.asin(x)
    return _torch.asin(x)


def acos(x):
    if isinstance(x, float):
        return math.acos(x)
    return _torch.acos(x)


def atan(x):
    if isinstance(x, float):
        return math.atan(x)
    return _torch.atan(x)


def atan2(x, y):
    if isinstance(x, float):
        return math.atan2(x, y)
    return _torch.atan2(x, y)


def sinh(x):
    if isinstance(x, float):
        return math.sinh(x)
    return _torch.sinh(x)


def cosh(x):
    if isinstance(x, float):
        return math.cosh(x)
    return _torch.cosh(x)


def tanh(x):
    if isinstance(x, float):
        return math.tanh(x)
    return _torch.tanh(x)


def asinh(x):
    if isinstance(x, float):
        return math.asinh(x)
    return _torch.asinh(x)


def acosh(x):
    if isinstance(x, float):
        return math.acosh(x)
    return _torch.acosh(x)


def atanh(x):
    if isinstance(x, float):
        return math.atanh(x)
    return _torch.atanh(x)


def log(x):
    if isinstance(x, float):
        return math.log(x)
    return _torch.log(x)


def exp(x):
    if isinstance(x, float):
        return math.exp(x)
    return _torch.exp(x)


def erf(x):
    if isinstance(x, float):
        return math.erf(x)
    return _torch.erf(x)
