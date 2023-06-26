import torch
import ivy
from ivy.torch.core import Variable
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.torch import to_ivy_arrays_and_back
from ivy.functional.frontends.torch.tensor.math import tanh as torch_tanh


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def selu(
    x,
    /,
    *,
    alpha=1.6732632423543772848170429916717,
    scale=1.0507009873554804934193349852946,
    name=None,
):
    if scale <= 1.0:
        raise ValueError(f"The scale must be greater than 1.0. Received: {scale}.")

    if alpha < 0:
        raise ValueError(f"The alpha must be no less than zero. Received: {alpha}.")

    ret = ivy.where(x > 0, x, alpha * ivy.expm1(x))
    arr = scale * ret
    return ivy.astype(arr, x.dtype)


tanh = torch_tanh


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def hardshrink(x, threshold=0.5, name=None):
    mask = ivy.logical_or(ivy.greater(x, threshold), ivy.less(x, -threshold))
    return ivy.where(mask, x, 0.0)


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def hardtanh(
    x,
    /,
    *,
    min_val=-1.0,
    max_val=1.0,
    name=None,
):
    less = ivy.where(ivy.less(x, min_val), min_val, x)
    ret = ivy.where(ivy.greater(x, max_val), max_val, less).astype(x.dtype)
    return ret


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def gelu(x, approximate=False, name=None):
    return ivy.gelu(x, approximate=approximate)


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def hardsigmoid(x, slope=0.1666667, offset=0.5, name=None):
    ret = ivy.minimum(ivy.maximum(ivy.add(ivy.multiply(x, slope), offset), 0), 1)
    return ret


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def relu6(x, name=None):
    return ivy.relu6(x)


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def softshrink(
    x,
    /,
    *,
    threshold=0.5,
    name=None,
):
    low = ivy.where(ivy.less(x, -threshold), ivy.add(x, threshold),
    up = ivy.where(ivy.greater(x, threshold), ivy.subtract(x, threshold), 0)
    add = ivy.add(low, up)
    return ivy.astype(add, x.dtype)


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def softsign(
    x,
    /,
    *,
    name=None,
):
    return ivy.divide(x, ivy.add(1, ivy.abs(x)))


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def hardshrink(x, threshold=0.5, name=None):
    low = ivy.where(ivy.less(x, -threshold), x, 0.0)
    up = ivy.where(ivy.greater(x, threshold), x, 0.0)
    add = ivy.add(low, up)
    return ivy.astype(add, x.dtype)


@with_supported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def cosine_similarity(x1, x2, *, axis=1, eps=1e-08):
    if len(x1.shape) == len(x2.shape) and len(x2.shape) >= 2:
        numerator = ivy.sum(x1 * x2, axis=axis)
        x1_squared_norm = ivy.sum(ivy.square(x1), axis=axis)
        x2_squared_norm = ivy.sum(ivy.square(x2), axis=axis)
    else:
        numerator = ivy.sum(x1 * x2)
        x1_squared_norm = ivy.sum(ivy.square(x1))
        x2_squared_norm = ivy.sum(ivy.square(x2))

    x1_norm = ivy.sqrt(x1_squared_norm)
    x2_norm = ivy.sqrt(x2_squared_norm)
    norm_mm = x1_norm * x2_norm
    denominator = ivy.maximum(norm_mm, eps)

    cosine = numerator / denominator
    return cosine
