from typing import Optional, Union

# global
import numpy as np
import ivy

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
) -> np.ndarray:
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
    if not ivy.is_array(x):
        return np.asarray(1 / (1 + np.exp(-x)))
    return np.asarray(1 / (1 + np.exp(-x))).astype(x.dtype)


def hard_tanh(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.where(x > 1, 1, np.where(x < -1, -1, x)).astype(x.dtype)


@_scalar_output_to_0d_array
def softplus(
    x: np.ndarray,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:

    if beta is not None and beta != 1:
        x_beta = x * beta
        res = (
            np.add(
                np.log1p(np.exp(-np.abs(x_beta))),
                np.maximum(x_beta, 0, dtype=x.dtype),
                out=out,
            )
        ) / beta
    else:
        x_beta = x
        res = np.add(
            np.log1p(np.exp(-np.abs(x_beta))),
            np.maximum(x_beta, 0, dtype=x.dtype),
            out=out,
        )
    if threshold is not None:
        return np.where(x_beta > threshold, x, res).astype(x.dtype)
    return res.astype(x.dtype)


def softsign(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.divide(x, np.abs(x) + 1, out=out)


def silu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.multiply(x, sigmoid(x), out=out)


def log_sigmoid(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return -softplus(-x, out=out)


@_scalar_output_to_0d_array
def hard_sigmoid(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    res = relu6(x + 3.0, out=out) / 6.0
    return res.astype(x.dtype)


def hard_silu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.multiply(x, hard_sigmoid(x), out=out)


def elu(
    x: np.ndarray, /, *, alpha: float = 1.0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.where(x > 0, x, alpha * np.expm1(x), out=out).astype(x.dtype)


def selu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    alpha = 1.6732632423543772848170429916717
    scale = np.array(1.0507009873554804934193349852946, dtype=x.dtype)
    return np.multiply(scale, elu(x, alpha=alpha), out=out)


def celu(
    x: np.ndarray, /, *, alpha: float = 1.0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.where(x > 0, x, alpha * np.expm1(x / alpha), out=out).astype(x.dtype)


def glu(
    x: np.ndarray, /, *, axis: int = -1, out: Optional[np.ndarray] = None
) -> np.ndarray:
    size = x.shape[axis]
    assert size % 2 == 0, "axis size must be divisible by 2"
    x1, x2 = np.split(x, 2, axis)
    return np.divide(x1, 1 + np.exp(-x2), out=out)
