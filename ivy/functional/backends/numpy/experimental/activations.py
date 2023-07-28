from typing import Optional, Union

# global
import numpy as np

# local
import ivy
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


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


@with_unsupported_dtypes({"1.25.1 and below": ("bool",)}, backend_version)
@_scalar_output_to_0d_array
def logsigmoid(input: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
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
