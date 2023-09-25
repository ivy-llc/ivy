# local
from ..math import *  # noqa: F401
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back

# NOTE:
# Only inplace functions are to be added in this file.
# Please add non-inplace counterparts to `/frontends/paddle/math.py`.


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def abs(x, name=None):
    return ivy.abs(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def acos(x, name=None):
    return ivy.acos(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def acosh(x, name=None):
    return ivy.acosh(x)


@with_unsupported_dtypes(
    {"2.5.1 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")}, "paddle"
)
@to_ivy_arrays_and_back
def add(x, y, name=None):
    return ivy.add(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def addmm(input, x, y, beta=1.0, alpha=1.0, name=None):
    value = alpha * ivy.matmul(x, y) + (beta * input)
    return value


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def amax(x, axis=None, keepdims=False):
    if axis is None:
        return ivy.max(x)
    if isinstance(axis, int):
        axis = [axis]
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += x.ndim
    for i in axis:
        if i < 0 or i >= x.ndim:
            raise ValueError("axis {} is out of range [-{}:{}]".format(i, 0, x.ndim))
    return ivy.max(x, axis=axis, keepdims=keepdims)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def amin(x, axis=None, keepdim=False, name=None):
    return ivy.min(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.1 and below": ("complex64", "complex128", "float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def angle(x, name=None):
    return ivy.angle(x)


@with_supported_dtypes({"2.5.0 and below": "bool"}, "paddle")
@to_ivy_arrays_and_back
def any(x, axis=None, keepdim=False, name=None):
    return ivy.any(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def asin(x, name=None):
    return ivy.asin(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def asinh(x, name=None):
    return ivy.asinh(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def atan(x, name=None):
    return ivy.atan(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def atan2(x, y, name=None):
    return ivy.atan2(x, y)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def atanh(x, name=None):
    return ivy.atanh(x)


def broadcast_shape(x_shape, y_shape):
    return list(ivy.broadcast_shapes(x_shape, y_shape))


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def ceil(x, name=None):
    return ivy.ceil(x)


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
def floor_(x, name=None):
    return ivy.inplace_update(x, floor(x))


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
def sqrt_(x, name=None):
    return ivy.inplace_update(x, sqrt(x))


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def subtract_(x, y, name=None):
    return ivy.inplace_update(x, subtract(x, y))


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def tanh_(x, name=None):
    return ivy.inplace_update(x, tanh(x))
