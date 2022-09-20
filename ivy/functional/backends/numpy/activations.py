"""Collection of Numpy activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional, Union

# global
import numpy as np

import ivy
from ivy.functional.backends.numpy.helpers import _handle_0_dim_output

try:
    from scipy.special import erf
except (ImportError, ModuleNotFoundError):
    erf = None


def relu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.asarray(np.maximum(x, 0, out=out, dtype=x.dtype))


relu.support_native_out = True


def leaky_relu(
    x: np.ndarray, /, *, alpha: float = 0.2, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.asarray(np.where(x > 0, x, np.multiply(x, alpha)), x.dtype)


def gelu(
    x, /, *, approximate: bool = True, out: Optional[np.ndarray] = None
) -> np.ndarray:
    ivy.assertions.check_exists(
        erf,
        message="scipy must be installed in order to call ivy.gelu with a \
        numpy backend.",
    )
    if approximate:
        ret = 0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
    else:
        ret = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return np.asarray(ret.astype(x.dtype))


def sigmoid(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if not ivy.is_array(x):
        return np.asarray(1 / (1 + np.exp(-x)))
    return np.asarray(1 / (1 + np.exp(-x))).astype(x.dtype)


def softmax(
    x: np.ndarray, /, *, axis: Optional[int] = None, out: Optional[np.ndarray] = None
) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return np.divide(exp_x, np.sum(exp_x, axis=axis, keepdims=True), out=out)


softmax.support_native_out = True


@_handle_0_dim_output
def softplus(x: np.ndarray,
             /,
             *,
             beta: Optional[Union[int, float]] = None,
             threshold: Optional[Union[int, float]] = None,
             out: Optional[np.ndarray] = None
             ) -> np.ndarray:

    if beta is not None and beta != 1:
        x_beta = x * beta
        res = (np.add(
            np.log1p(np.exp(-np.abs(x_beta))),
            np.maximum(x_beta, 0, dtype=x.dtype),
            out=out
        )) / beta
    else:
        x_beta = x
        res = (np.add(
            np.log1p(np.exp(-np.abs(x_beta))),
            np.maximum(x_beta, 0, dtype=x.dtype),
            out=out
        ))
    if threshold is not None:
        return np.where(x_beta > threshold, x, res)
    return res


softplus.support_native_out = True
