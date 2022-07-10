"""Collection of Numpy activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional

# global
import numpy as np


try:
    from scipy.special import erf
except (ImportError, ModuleNotFoundError):
    erf = None


def relu(
    x: np.ndarray, 
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.maximum(x, 0, out=out)


relu.support_native_out = True


def leaky_relu(
    x: np.ndarray, 
    alpha: Optional[float] = 0.2,
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.where(x > 0, x, x * alpha)


def gelu(
    x, 
    approximate: Optional[bool] = True,
    *,
    out: Optional[np.ndarray] = None
):
    if erf is None:
        raise Exception(
            "scipy must be installed in order to call ivy.gelu with a numpy backend."
        )
    if approximate:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def sigmoid(
    x: np.ndarray,
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return 1 / (1 + np.exp(-x, out=out))


sigmoid.support_native_out = True


def tanh(
    x: np.ndarray, 
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return (np.exp(x, out=out) - np.exp(-x, out=out)) / (
        np.exp(x, out=out) + np.exp(-x, out=out)
    )


tanh.support_native_out = True


def softmax(
    x: np.ndarray, 
    axis: Optional[int] = None, 
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    exp_x = np.exp(x, out=out)
    return exp_x / np.sum(exp_x, axis, keepdims=True, out=out)


softmax.support_native_out = True


def softplus(
    x: np.ndarray, 
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x, out=out), out=out), out=out) + np.maximum(
        x, 0, out=out
    )


softplus.support_native_out = True
