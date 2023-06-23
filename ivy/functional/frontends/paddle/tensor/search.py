# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_ivy_arrays_and_back
def argmax(x, /, *, axis=None, keepdim=False, dtype="int64", name=None):
    return ivy.argmax(x, axis=axis, keepdims=keepdim, dtype=dtype)


@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_ivy_arrays_and_back
def argmin(x, /, *, axis=None, keepdim=False, dtype="int64", name=None):
    return ivy.argmin(x, axis=axis, keepdims=keepdim, dtype=dtype)
