from typing import Optional, Union, Literal

# global
import numpy as np

# local
import ivy
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import (
    with_unsupported_dtypes,
)
from . import backend_version


def logit(
    x: np.ndarray,
    /,
    *,
    eps: Optional[float] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
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
def relu6(
    x: np.ndarray, /, *, complex_mode="jax", out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.minimum(np.maximum(x, 0, dtype=x.dtype), 6, out=out, dtype=x.dtype)


relu6.support_native_out = True


@with_unsupported_dtypes({"1.26.3 and below": ("bool",)}, backend_version)
@_scalar_output_to_0d_array
def logsigmoid(
    input: np.ndarray, /, *, complex_mode="jax", out: Optional[np.ndarray] = None
) -> np.ndarray:
    return -(np.log1p(np.exp(-(input))))


@_scalar_output_to_0d_array
def selu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    ret = (scale * np.where(x > 0, x, alpha * np.expm1(x))).astype(x.dtype)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ret


selu.support_native_out = True


@_scalar_output_to_0d_array
def silu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    ret = np.asarray(x * (1 / (1 + np.exp(-x))))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    if not ivy.is_array(x):
        return ret
    else:
        return np.asarray(x * (1 / (1 + np.exp(-x)))).astype(x.dtype)


silu.support_native_out = True


@_scalar_output_to_0d_array
def elu(
    x: np.ndarray, /, *, alpha: float = 1.0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    # exp = np.expm1(x)
    ret = np.where(x > 0, x, np.multiply(alpha, np.expm1(x))).astype(x.dtype)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ret


elu.support_native_out = True


@_scalar_output_to_0d_array
def celu(
    x: np.ndarray,
    /,
    *,
    alpha: float = 1.0,
    complex_mode="jax",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return (np.maximum(0, x) + alpha * np.expm1(np.minimum(0, x) / alpha)).astype(
        x.dtype
    )


@with_unsupported_dtypes({"1.25.2 and below": ("float16", "bfloat16")}, backend_version)
@_scalar_output_to_0d_array
def hardtanh(
    x: np.ndarray,
    /,
    *,
    max_val: float = 1.0,
    min_val: float = -1.0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.where(x > max_val, max_val, np.where(x < min_val, min_val, x))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


hardtanh.support_native_out = True


@_scalar_output_to_0d_array
def tanhshrink(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    ret = np.subtract(x, np.tanh(x))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


tanhshrink.support_native_out = True


@_scalar_output_to_0d_array
def threshold(
    x: np.ndarray,
    /,
    *,
    threshold: float,
    value: float,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.where(x > threshold, x, value)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


threshold.support_native_out = True


@_scalar_output_to_0d_array
def softshrink(
    x: np.ndarray, /, *, lambd: float = 0.5, out: Optional[np.ndarray] = None
) -> np.ndarray:
    ret = np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


softshrink.support_native_out = True


@_scalar_output_to_0d_array
def scaled_tanh(
    x: np.ndarray,
    /,
    *,
    alpha: float = 1.7159,
    beta: float = 0.67,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return alpha * np.tanh(beta * x)


@_scalar_output_to_0d_array
def hardshrink(
    x: np.ndarray, /, *, lambd: float = 0.5, out: Optional[np.ndarray] = None
) -> np.ndarray:
    ret = np.where(x > lambd, x, np.where(x < -lambd, x, 0))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


hardshrink.support_native_out = True


@with_unsupported_dtypes({"2.14.0 and below": ("complex",)}, backend_version)
@_scalar_output_to_0d_array
def hardsilu(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    ret = x * np.divide(relu6(x + 3), 6)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ivy.astype(ret, x.dtype)


hardsilu.support_native_out = True
