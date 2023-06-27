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


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
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
    return ivy.sign(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def neg(x, name=None):
    return ivy.negative(x)


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def exp(x, name=None):
    return ivy.exp(x)


@with_supported_dtypes(
    {
        "2.4.2 and below": (
            "float32",
            "float64",
            "int32",
            "int64",
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
