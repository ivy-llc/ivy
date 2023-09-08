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



@with_unsupported_dtypes({"2.4.2 and below": ("int16", "float16")}, "paddle")
@to_ivy_arrays_and_back
def conj(x, name=None):
    return ivy.conj(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def cos(x, name=None):
    return ivy.cos(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def cosh(x, name=None):
    return ivy.cosh(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("int32", "int64", "float16", "float32", "float64", "bool")},
    "paddle",
)
@to_ivy_arrays_and_back
def count_nonzero(x, axis=None, dtype=None, keepdims=False, name=None):
    return ivy.astype(
        ivy.count_nonzero(x, axis=axis, dtype=dtype, keepdims=keepdims), ivy.int64
    )


@with_supported_dtypes(
    {
        "2.5.1 and below": (
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


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def deg2rad(x, name=None):
    return ivy.deg2rad(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def diff(x, n=1, axis=-1, prepend=None, append=None, name=None):
    return ivy.diff(x, n=n, axis=axis, prepend=prepend, append=append)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def digamma(x, name=None):
    digamma_fun = ivy.digamma
    return ivy.array(digamma_fun(x), dtype=x.dtype)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def divide(x, y, name=None):
    return ivy.divide(x, y)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def erf(x, name=None):
    return ivy.erf(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def exp(x, name=None):
    return ivy.exp(x)



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
def sqrt_(x, name=None):
    return ivy.inplace_update(x, sqrt(x))
