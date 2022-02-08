"""
Collection of Numpy activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np
try:
    from scipy.special import erf as _erf
except (ImportError, ModuleNotFoundError):
    _erf = None

relu = lambda x: _np.maximum(x, 0)
leaky_relu = lambda x, alpha=0.2: _np.where(x > 0, x, x * alpha)


def gelu(x, approximate=True):
    if _erf is None:
        raise Exception('scipy must be installed in order to call ivy.gelu with a numpy backend.')
    if approximate:
        return 0.5 * x * (1 + _np.tanh(_np.sqrt(2 / _np.pi) * (x + 0.044715 * x ** 3)))
    return 0.5 * x * (1 + _erf(x/_np.sqrt(2)))


tanh = _np.tanh
sigmoid = lambda x: 1 / (1 + _np.exp(-x))


def softmax(x, axis=-1):
    exp_x = _np.exp(x)
    return exp_x / _np.sum(exp_x, axis, keepdims=True)


softplus = lambda x: _np.log(_np.exp(x) + 1)
