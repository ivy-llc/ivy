# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
from ivy.functional.frontends.paddle.tensor.math import (
    log_softmax as paddle_log_softmax,
)


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
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


tanh = paddle_tanh
log_softmax = paddle_log_softmax


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def hardshrink(x, threshold=0.5, name=None):
    mask = ivy.logical_or(ivy.greater(x, threshold), ivy.less(x, -threshold))
    return ivy.where(mask, x, 0.0)


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def hardtanh(
    x,
    /,
    *,
    min=-1.0,
    max=1.0,
    name=None,
):
    less = ivy.where(ivy.less(x, min), min, x)
    ret = ivy.where(ivy.greater(x, max), max, less).astype(x.dtype)
    return ret


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def gelu(x, approximate=False, name=None):
    return ivy.gelu(x, approximate=approximate)


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def hardsigmoid(x, slope=0.1666667, offset=0.5, name=None):
    ret = ivy.minimum(ivy.maximum(ivy.add(ivy.multiply(x, slope), offset), 0), 1)
    return ret
