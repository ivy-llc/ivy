# global
import ivy
import ivy.functional.frontends.paddle as paddle
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    handle_out_argument,
    with_supported_dtypes,
)
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("uint8", "int8", "int16", "complex64", "complex128")}, "paddle"
)
@to_ivy_arrays_and_back
def equal(x, y, /, *, name=None):
    return ivy.equal(x, y)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("uint8", "int8", "int16", "complex64", "complex128")}, "paddle"
)
@to_ivy_arrays_and_back
def not_equal(x, y, /, *, name=None):
    return ivy.not_equal(x, y)


@with_unsupported_dtypes(
    {
        "2.5.0 and below": (
            "uint8",
            "int8",
            "int16",
            "float16",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def equal_all(x, y, /, *, name=None):
    return paddle.to_tensor([ivy.array_equal(x, y)])


@with_unsupported_dtypes(
    {"2.5.0 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def greater_than(x, y, /, *, name=None):
    return ivy.greater(x, y)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def greater_equal(x, y, /, *, name=None):
    return ivy.greater_equal(x, y)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def less_than(x, y, /, *, name=None):
    return ivy.less(x, y)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def less_equal(x, y, /, *, name=None):
    return ivy.less_equal(x, y)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
@handle_out_argument
def logical_or(x, y, /, *, name=None, out=None):
    return ivy.logical_or(x, y, out=out)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
@handle_out_argument
def logical_xor(x, y, /, *, name=None, out=None):
    return ivy.logical_xor(x, y, out=out)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
@handle_out_argument
def logical_not(x, /, *, name=None, out=None):
    return ivy.logical_not(x, out=out)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
@handle_out_argument
def bitwise_or(x, y, name=None, out=None):
    return ivy.bitwise_or(x, y, out=out)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
@handle_out_argument
def bitwise_and(x, y, /, *, name=None, out=None):
    return ivy.bitwise_and(x, y, out=out)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
@handle_out_argument
def bitwise_xor(x, y, /, *, name=None, out=None):
    return ivy.bitwise_xor(x, y, out=out)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
@handle_out_argument
def bitwise_not(x, out=None, name=None):
    return ivy.bitwise_invert(x, out=out)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "float32",
            "float64",
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
@handle_out_argument
def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
    ret = ivy.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return paddle.to_tensor([ret])


@to_ivy_arrays_and_back
def is_tensor(x):
    return ivy.is_array(x)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
    return ivy.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
@handle_out_argument
def logical_and(x, y, /, *, name=None, out=None):
    return ivy.logical_and(x, y, out=out)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("uint8", "int8", "int16", "complex64", "complex128")}, "paddle"
)
@to_ivy_arrays_and_back
def is_empty(x, name=None):
    return ivy.is_empty(x)
