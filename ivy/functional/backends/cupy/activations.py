"""Collection of Numpy activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional, Union

# global
import cupy as cp

import ivy
from ivy.functional.backends.cupy.helpers import _handle_0_dim_output

try:
    from scipy.special import erf
except (ImportError, ModuleNotFoundError):
    erf = None


def relu(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.asarray(cp.maximum(x, 0, out=out, dtype=x.dtype))


relu.support_native_out = True


def leaky_relu(
    x: cp.ndarray, /, *, alpha: float = 0.2, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.asarray(cp.where(x > 0, x, cp.multiply(x, alpha)), x.dtype)


def gelu(
    x, /, *, approximate: bool = True, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    ivy.assertions.check_exists(
        erf,
        message="scipy must be installed in order to call ivy.gelu with a \
        numpy backend.",
    )
    if approximate:
        ret = 0.5 * x * (1 + cp.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
    else:
        ret = 0.5 * x * (1 + erf(x / cp.sqrt(2)))
    return cp.asarray(ret.astype(x.dtype))


def sigmoid(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    if not ivy.is_array(x):
        return cp.asarray(1 / (1 + cp.exp(-x)))
    return cp.asarray(1 / (1 + cp.exp(-x))).astype(x.dtype)


def softmax(
    x: cp.ndarray, /, *, axis: Optional[int] = None, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    exp_x = cp.exp(x - cp.max(x, axis=axis, keepdims=True))
    return cp.divide(exp_x, cp.sum(exp_x, axis=axis, keepdims=True), out=out)


softmax.support_native_out = True


@_handle_0_dim_output
def softplus(
    x: cp.ndarray,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:

    if beta is not None and beta != 1:
        x_beta = x * beta
        res = (
            cp.add(
                cp.log1p(cp.exp(-cp.abs(x_beta))),
                cp.maximum(x_beta, 0, dtype=x.dtype),
                out=out,
            )
        ) / beta
    else:
        x_beta = x
        res = cp.add(
            cp.log1p(cp.exp(-cp.abs(x_beta))),
            cp.maximum(x_beta, 0, dtype=x.dtype),
            out=out,
        )
    if threshold is not None:
        return cp.where(x_beta > threshold, x, res)
    return res


softplus.support_native_out = True


@_handle_0_dim_output
def log_softmax(
    x: cp.ndarray, /, *, axis: Optional[int] = None, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    x_max = cp.max(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~cp.isfinite(x_max)] = 0
    elif not cp.isfinite(x_max):
        x_max = 0
    exp_tmp = cp.exp(x - x_max)

    with cp.errstate(divide="ignore"):
        s = cp.sum(exp_tmp, axis=axis, keepdims=True)
        ret = cp.log(s)

    ret = x - x_max - ret
    return ret


log_softmax.support_native_out = True
