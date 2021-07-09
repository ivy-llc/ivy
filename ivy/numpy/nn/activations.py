"""
Collection of Numpy activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np

relu = lambda x: _np.maximum(x, 0)
leaky_relu = lambda x, alpha=0.2: _np.where(x > 0, x, x * alpha)
tanh = _np.tanh
sigmoid = lambda x: 1 / (1 + _np.exp(-x))


def softmax(x, axis=-1):
    exp_x = _np.exp(x)
    return exp_x / _np.sum(exp_x, axis, keepdims=True)


softplus = lambda x: _np.log(_np.exp(x) + 1)
