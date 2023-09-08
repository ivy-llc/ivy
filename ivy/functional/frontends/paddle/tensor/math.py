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

def erf(x, name=None):
    return ivy.erf(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def exp(x, name=None):
    return ivy.exp(x)


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
def lgamma(x, name=None):
    return ivy.lgamma(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def log(x, name=None):
    return ivy.log(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def log1p(x, name=None):
    return ivy.log1p(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def log2(x, name=None):
    return ivy.log2(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def logit(x, eps=None, name=None):
    return ivy.logit(x, eps=eps)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def max(x, axis=None, keepdim=False, name=None):
    return ivy.max(x, axis=axis, keepdims=keepdim)


# maximum
@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def maximum(x, y, name=None):
    return ivy.maximum(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def min(x, axis=None, keepdim=False, name=None):
    return ivy.min(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def minimum(x, y, name=None):
    return ivy.minimum(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def mm(input, mat2, name=None):
    return ivy.matmul(input, mat2)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def multiply(x, y, name=None):
    return ivy.multiply(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int8", "int16", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def neg(x, name=None):
    return ivy.negative(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def outer(x, y, name=None):
    return ivy.outer(x, y)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def pow(x, y, name=None):
    return ivy.pow(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def prod(x, axis=None, keepdim=False, dtype=None, name=None):
    return ivy.prod(x, axis=axis, keepdims=keepdim, dtype=dtype)



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
