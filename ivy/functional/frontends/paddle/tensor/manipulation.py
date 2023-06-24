# global
import ivy
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
def reshape(x, shape):
    return ivy.reshape(x, shape)


@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def abs(x, name=None):
    return ivy.abs(x)


absolute = abs


@to_ivy_arrays_and_back
def stack(x, axis=0, name=None):
    return ivy.stack(x, axis=axis)


@with_unsupported_dtypes({"2.5.0 and below": ("int8", "int16")}, "paddle")
@to_ivy_arrays_and_back
def concat(x, axis, name=None):
    return ivy.concat(x, axis=axis)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("int8", "uint8", "int16", "float16")},
    "paddle",
)
@to_ivy_arrays_and_back
def tile(x, repeat_times, name=None):
    return ivy.tile(x, repeats=repeat_times)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def split(x, num_or_sections, axis=0, name=None):
    return ivy.split(x, num_or_size_splits=num_or_sections, axis=axis)
