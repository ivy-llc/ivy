"""Collection of Numpy activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional

# global
import numpy as np


try:
    from scipy.special import erf
except (ImportError, ModuleNotFoundError):
    erf = None


def relu(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.asarray(np.maximum(x, 0, out=out, dtype=x.dtype))


def leaky_relu(
    x: np.ndarray,
    alpha: Optional[float] = 0.2,
) -> np.ndarray:
    return np.asarray(np.where(x > 0, x, x * alpha), x.dtype)


def gelu(x, approximate: Optional[bool] = True) -> np.ndarray:
    if erf is None:
        raise Exception(
            "scipy must be installed in order to call ivy.gelu with a numpy backend."
        )
    if approximate:
        ret = 0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
    else:
        ret = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return np.asarray(ret.astype(x.dtype))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-x))).astype(x.dtype)


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(
    x: np.ndarray,
    axis: Optional[int] = None,
) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis, keepdims=True)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
