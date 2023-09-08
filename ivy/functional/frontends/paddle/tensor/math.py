# local
from ..math import *  # noqa: F401
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def ceil_(x, name=None):
    return ivy.ceil(x, out=x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def exp_(x, name=None):
    return ivy.inplace_update(x, exp(x))


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def lerp_(x, y, weight, name=None):
    return ivy.inplace_update(x, lerp(x, y, weight))


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def reciprocal_(x, name=None):
    return ivy.inplace_update(x, reciprocal(x))


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def round_(x, name=None):
    return ivy.inplace_update(x, round(x))


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def rsqrt_(x, name=None):
    return ivy.inplace_update(x, reciprocal(sqrt(x)))


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def sinh(x, name=None):
    return ivy.sinh(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def sqrt(x, name=None):
    return ivy.sqrt(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def square(x, name=None):
    return ivy.square(x)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def stanh(x, scale_a=0.67, scale_b=1.7159, name=None):
    # TODO this function will be simplified as soon as the ivy.stanh(x,a,b) is added
    exp_ax = ivy.exp(ivy.multiply(scale_a, x))
    exp_minus_ax = ivy.exp(ivy.multiply(-scale_a, x))
    numerator = ivy.subtract(exp_ax, exp_minus_ax)
    denominator = ivy.add(exp_ax, exp_minus_ax)
    ret = ivy.multiply(scale_b, ivy.divide(numerator, denominator))
    return ret


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def subtract(x, y, name=None):
    return ivy.subtract(x, y)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def subtract_(x, y, name=None):
    return ivy.inplace_update(x, subtract(x, y))


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int6")}, "paddle"
)
@to_ivy_arrays_and_back
def take(
    x,
    index,
    mode="raise",
    name=None,
):
    if mode not in ["raise", "wrap", "clip"]:
        raise ValueError(
            "'mode' in 'take' should be 'raise', 'wrap', 'clip', but received {}."
            .format(mode)
        )
    x = ivy.reshape(x, (-1,))
    if mode == "clip":
        index = ivy.clip(index, 0, x.shape[-1] - 1)
    elif mode == "wrap":
        index = ivy.where(index < 0, index % x.shape[-1], index)
        index = ivy.where(index >= x.shape[-1], index % x.shape[-1], index)
    return ivy.gather(x, index, axis=0)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def tan(x, name=None):
    return ivy.tan(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def tanh(x, name=None):
    return ivy.tanh(x)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def trunc(x, name=None):
    return ivy.trunc(x)

