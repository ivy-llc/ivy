from typing import Optional, Union

# global
import numpy as np

# local
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array


def logit(
    x: np.ndarray,
    /,
    *,
    eps: Optional[float] = None,
    out: Optional[np.ndarray] = None,
):
    x_dtype = x.dtype
    if eps is None:
        x = np.where(np.logical_or(x > 1, x < 0), np.nan, x)
    else:
        x = np.clip(x, eps, 1 - eps)
    ret = (np.log(x / (1 - x))).astype(x_dtype)
    if np.isscalar(ret):
        return np.array(ret)
    return ret


def hardshrink(
    x: np.ndarray,
    /,
    *,
    lambd: Optional[float] = 0.5,
    out: Optional[np.ndarray] = None,
):
    mask = np.logical_or(np.greater(x, lambd), np.less(x, -lambd))
    return np.where(mask, x, 0.0).astype(x.dtype)


def softshrink(
    x: np.ndarray,
    /,
    *,
    lambd: Optional[float] = 0.5,
    out: Optional[np.ndarray] = None,
):
    low = np.where(np.less(x, -lambd), np.add(x, lambd), 0)
    up = np.where(np.greater(x, lambd), np.subtract(x, lambd), 0)
    return np.add(low, up, dtype=x.dtype, out=out)


@_scalar_output_to_0d_array
def thresholded_relu(
    x: np.ndarray,
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.where(x > threshold, x, 0).astype(x.dtype)


thresholded_relu.support_native_out = True


@_scalar_output_to_0d_array
def threshold(
    x: np.ndarray,
    threshold: Union[int, float],
    value: Union[int, float],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.where(x > threshold, x, value).astype(x.dtype)


thresholded_relu.support_native_out = True


@_scalar_output_to_0d_array
def relu6(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.minimum(np.maximum(x, 0, dtype=x.dtype), 6, out=out, dtype=x.dtype)


relu6.support_native_out = True


def batch_norm(
    x: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    /,
    *,
    scale: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    training: bool = False,
    eps: float = 1e-5,
):
    ndims = len(x.shape)
    if training:
        dims = (0, *range(2, ndims))
        mean = np.mean(x, axis=dims)
        variance = np.var(x, axis=dims)
    x = np.transpose(x, (0, *range(2, ndims), 1))
    inv = 1.0 / np.sqrt(variance + eps)
    if scale is not None:
        inv *= scale
    ret = x * inv.astype(x.dtype, copy=False) + (
        offset - mean * inv if offset is not None else -mean * inv
    ).astype(x.dtype)
    return np.transpose(ret, (0, ndims - 1, *range(1, ndims - 1)))


def sigmoid(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.divide(1, 1 + np.exp(-x), out=out).astype(x.dtype)


@_scalar_output_to_0d_array
def logsigmoid(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return -np.logaddexp(-x, 0, out=out).astype(x.dtype)


def hard_tanh(
    x: np.ndarray,
    /,
    *,
    min_value: float = -1.0,
    max_value: float = 1.0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.clip(x, a_min=min_value, a_max=max_value, out=out).astype(x.dtype)


def softsign(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.divide(x, np.abs(x) + 1, out=out)


def silu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.multiply(x, sigmoid(x), out=out)


@_scalar_output_to_0d_array
def hard_sigmoid(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    res = relu6(x + 3.0, out=out) / 6.0
    return res.astype(x.dtype)


def hard_silu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.multiply(x, hard_sigmoid(x), out=out)


def elu(
    x: np.ndarray, /, *, alpha: float = 1.0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.where(x > 0, x, alpha * np.expm1(x)).astype(x.dtype)


def parametric_relu(
    x: np.ndarray,
    weight: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.where(x >= 0, x, weight * x).astype(x.dtype)


def selu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    alpha = 1.6732632423543772848170429916717
    scale = np.array(1.0507009873554804934193349852946, dtype=x.dtype)
    return np.multiply(scale, elu(x, alpha=alpha), out=out)


def celu(
    x: np.ndarray, /, *, alpha: float = 1.0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.where(x > 0, x, alpha * np.expm1(x / alpha)).astype(x.dtype)


def glu(
    x: np.ndarray, /, *, axis: int = -1, out: Optional[np.ndarray] = None
) -> np.ndarray:
    size = x.shape[axis]
    assert size % 2 == 0, "axis size must be divisible by 2"
    x1, x2 = np.split(x, 2, axis)
    return np.divide(x1, 1 + np.exp(-x2), out=out)
