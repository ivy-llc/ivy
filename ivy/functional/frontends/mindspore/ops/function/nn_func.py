"""Includes Mindspore Frontend functions listed in the TODO list
https://github.com/unifyai/ivy/issues/14951."""


# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)

@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def softmax(x, name=None):
    return ivy.softmax(x, /, *, axis=None, out=None)