# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def imag(x):
    return ivy.imag(x)


@to_ivy_arrays_and_back
def is_complex(x):
    return ivy.is_complex_dtype(x)


@to_ivy_arrays_and_back
def is_floating_point(x):
    return ivy.is_float_dtype(x)


@to_ivy_arrays_and_back
def is_integer(x):
    return ivy.is_int_dtype(x)


@to_ivy_arrays_and_back
def rank(input):
    return ivy.get_num_dims(input)


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def real(x):
    return ivy.real(x)
