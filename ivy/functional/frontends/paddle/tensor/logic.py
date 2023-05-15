# global
import ivy
import ivy.functional.frontends.paddle as paddle
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("uint8", "int8", "int16", "complex64", "complex128")}, "paddle"
)
@to_ivy_arrays_and_back
def equal(x, y, /, *, name=None):
    return ivy.equal(x, y)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("uint8", "int8", "int16", "complex64", "complex128")}, "paddle"
)
@to_ivy_arrays_and_back
def not_equal(x, y, /, *, name=None):
    return ivy.not_equal(x, y)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
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
    {"2.4.2 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def greater_than(x, y, /, *, name=None):
    return ivy.greater(x, y)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def greater_equal(x, y, /, *, name=None):
    return ivy.greater_equal(x, y)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def less_than(x, y, /, *, name=None):
    return ivy.less(x, y)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def less_equal(x, y, /, *, name=None):
    return ivy.less_equal(x, y)
