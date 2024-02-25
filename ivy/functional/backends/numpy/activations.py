"""Collection of Numpy activation functions, wrapped to fit Ivy syntax and
signature."""

# global
from typing import Optional, Union, Literal
import numpy as np

# local
import ivy
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version


@with_supported_dtypes(
    {
        "1.26.3 and below": (
            "float",
            "int",
            "complex",
        )
    },
    backend_version,
)
@_scalar_output_to_0d_array
def relu(
    x: np.ndarray, /, *, complex_mode="jax", out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.maximum(x, 0, out=out, dtype=x.dtype)


relu.support_native_out = True


def leaky_relu(
    x: np.ndarray,
    /,
    *,
    alpha: float = 0.2,
    complex_mode="jax",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.asarray(np.where(x > 0, x, np.multiply(x, alpha)), x.dtype)


@_scalar_output_to_0d_array
def gelu(
    x: np.ndarray,
    /,
    *,
    approximate: bool = False,
    complex_mode="jax",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if approximate:
        ret = 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    else:
        ret = 0.5 * x * (1 + ivy.erf(x / np.sqrt(2)))
    return ivy.astype(ret, x.dtype, copy=False)


def sigmoid(
    x: np.ndarray, /, *, complex_mode="jax", out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not ivy.is_array(x):
        return np.asarray(1 / (1 + np.exp(-x)))
    return np.asarray(1 / (1 + np.exp(-x))).astype(x.dtype)


def softmax(
    x: np.ndarray, /, *, axis: Optional[int] = None, out: Optional[np.ndarray] = None
) -> np.ndarray:
    axis = -1 if axis is None else axis
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return np.divide(exp_x, np.sum(exp_x, axis=axis, keepdims=True), out=out)


softmax.support_native_out = True


@_scalar_output_to_0d_array
def softplus(
    x: np.ndarray,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    complex_mode="jax",
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


softplus.support_native_out = True


# Softsign
@_scalar_output_to_0d_array
def softsign(x: np.ndarray, /, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.divide(x, 1 + np.abs(x), out=out).astype(x.dtype)


softsign.support_native_out = True


@_scalar_output_to_0d_array
def log_softmax(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = -1,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    sub_tmp = np.subtract(x, x_max)
    ret = np.sum(np.exp(sub_tmp), axis=axis, keepdims=True)
    ret = np.log(ret)
    ret = np.subtract(sub_tmp, ret)
    return ret


log_softmax.support_native_out = True


@_scalar_output_to_0d_array
def mish(
    x: np.ndarray,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return x * np.tanh(np.log1p(np.exp(x)))


mish.support_native_out = True


@_scalar_output_to_0d_array
def hardswish(
    x: np.ndarray,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    max_x_3 = np.maximum(x + 3, 0, dtype=x.dtype)
    return (x * np.minimum(max_x_3, 6, out=out, dtype=x.dtype) / 6).astype(x.dtype)


hardswish.support_native_out = True
