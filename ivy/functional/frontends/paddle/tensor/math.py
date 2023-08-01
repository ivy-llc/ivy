# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def sin(x, name=None):
    return ivy.sin(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def cos(x, name=None):
    return ivy.cos(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def acos(x, name=None):
    return ivy.acos(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def cosh(x, name=None):
    return ivy.cosh(x)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def tanh(x, name=None):
    return ivy.tanh(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def acosh(x, name=None):
    return ivy.acosh(x)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def asin(x, name=None):
    return ivy.asin(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def log(x, name=None):
    return ivy.log(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def divide(x, y, name=None):
    return ivy.divide(x, y)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def abs(x, name=None):
    return ivy.abs(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def multiply(x, y, name=None):
    return ivy.multiply(x, y)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")}, "paddle"
)
@to_ivy_arrays_and_back
def add(x, y, name=None):
    return ivy.add(x, y)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def subtract(x, y, name=None):
    return ivy.subtract(x, y)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def sqrt(x, name=None):
    return ivy.sqrt(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def atanh(x, name=None):
    return ivy.atanh(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def atan(x, name=None):
    return ivy.atan(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def round(x, name=None):
    return ivy.round(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def ceil(x, name=None):
    return ivy.ceil(x)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def sinh(x, name=None):
    return ivy.sinh(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def pow(x, y, name=None):
    return ivy.pow(x, y)


@with_unsupported_dtypes({"2.4.2 and below": ("int16", "float16")}, "paddle")
@to_ivy_arrays_and_back
def conj(x, name=None):
    return ivy.conj(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def floor(x, name=None):
    return ivy.floor(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def remainder(x, y, name=None):
    return ivy.remainder(x, y)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def log2(x, name=None):
    return ivy.log2(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def log1p(x, name=None):
    return ivy.log1p(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def rad2deg(x, name=None):
    return ivy.rad2deg(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def deg2rad(x, name=None):
    return ivy.deg2rad(x)


@with_supported_dtypes({"2.5.0 and below": ("int32", "int64")}, "paddle")
@to_ivy_arrays_and_back
def gcd(x, y, name=None):
    return ivy.gcd(x, y)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def tan(x, name=None):
    return ivy.tan(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def atan2(x, y, name=None):
    return ivy.atan2(x, y)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def square(x, name=None):
    return ivy.square(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def sign(x, name=None):
    return ivy.sign(x, np_variant=False)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def neg(x, name=None):
    return ivy.negative(x)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def exp(x, name=None):
    return ivy.exp(x)


@with_supported_dtypes({"2.5.0 and below": ("float16", "float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def expm1(x, name=None):
    return ivy.expm1(x)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def erf(x, name=None):
    return ivy.erf(x)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def cumprod(x, dim=None, dtype=None, name=None):
    return ivy.cumprod(x, axis=dim, dtype=dtype)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def reciprocal(x, name=None):
    return ivy.reciprocal(x)


@with_supported_dtypes({"2.5.0 and below": ("int32", "int64")}, "paddle")
@to_ivy_arrays_and_back
def lcm(x, y, name=None):
    return ivy.lcm(x, y)


@with_supported_dtypes(
    {"2.5.0 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def isnan(x, name=None):
    return ivy.isnan(x)


@with_supported_dtypes(
    {"2.5.0 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def isfinite(x, name=None):
    return ivy.isfinite(x)


@with_supported_dtypes(
    {"2.5.0 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def isinf(x, name=None):
    return ivy.isinf(x)


@with_supported_dtypes(
    {"2.5.0 and below": ("complex64", "complex128", "float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def angle(x, name=None):
    return ivy.angle(x)


@with_unsupported_dtypes({"2.5.0 and below": "bfloat16"}, "paddle")
@to_ivy_arrays_and_back
def fmin(x, y, name=None):
    return ivy.fmin(x, y)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def logit(x, eps=None, name=None):
    return ivy.logit(x, eps=eps)


@with_unsupported_dtypes({"2.5.0 and below": "bfloat16"}, "paddle")
@to_ivy_arrays_and_back
def fmax(x, y, name=None):
    return ivy.fmax(x, y)


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def minimum(x, y, name=None):
    return ivy.minimum(x, y)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def trunc(x, name=None):
    return ivy.trunc(x)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def sgn(x, name=None):
    return ivy.sign(x, np_variant=True)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def outer(x, y, name=None):
    return ivy.outer(x, y)


# maximum
@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def maximum(x, y, name=None):
    return ivy.maximum(x, y)


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def frac(x, name=None):
    return x - ivy.sign(x) * ivy.floor(ivy.abs(x))


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def asinh(x, name=None):
    return ivy.asinh(x)


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def max(x, axis=None, keepdim=False, name=None):
    return ivy.max(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def lerp(x, y, weight, name=None):
    return ivy.lerp(x, y, weight)
