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
def ceil_(x, name=None):
    return ivy.ceil(x, out=x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def exp_(x, name=None):
    return ivy.inplace_update(x, exp(x))


@with_supported_dtypes({"2.5.1 and below": ("float16", "float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def expm1(x, name=None):
    return ivy.expm1(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("bfloat16", "float32", "float64")}, "paddle"
)
@to_ivy_arrays_and_back
def floor(x, name=None):
    return ivy.floor(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def floor_(x, name=None):
    return ivy.inplace_update(x, floor(x))


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def floor_divide(x, y, name=None):
    return ivy.floor_divide(x, y)


@with_unsupported_dtypes({"2.5.1 and below": "bfloat16"}, "paddle")
@to_ivy_arrays_and_back
def fmax(x, y, name=None):
    return ivy.fmax(x, y)


@with_unsupported_dtypes({"2.5.1 and below": "bfloat16"}, "paddle")
@to_ivy_arrays_and_back
def fmin(x, y, name=None):
    return ivy.fmin(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def frac(x, name=None):
    y = ivy.trunc(x)
    return ivy.subtract(x, y)


@with_supported_dtypes({"2.5.1 and below": ("int32", "int64")}, "paddle")
@to_ivy_arrays_and_back
def gcd(x, y, name=None):
    return ivy.gcd(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def heaviside(x, y, name=None):
    return ivy.heaviside(x, y)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def inner(x, y, name=None):
    result = ivy.inner(x, y)
    if (x.shape == () and y.shape == (1,)) or (x.shape == (1,) and y.shape == ()):
        result = result.reshape((1,))
    elif x.shape == (1,) and y.shape == (1,):
        result = result.reshape((1,))
    return result


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def isfinite(x, name=None):
    return ivy.isfinite(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def isinf(x, name=None):
    return ivy.isinf(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def isnan(x, name=None):
    return ivy.isnan(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def kron(x, y, name=None):
    return ivy.kron(x, y)


@with_supported_dtypes({"2.5.1 and below": ("int32", "int64")}, "paddle")
@to_ivy_arrays_and_back
def lcm(x, y, name=None):
    return ivy.lcm(x, y)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def lerp(x, y, weight, name=None):
    return ivy.lerp(x, y, weight)



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
