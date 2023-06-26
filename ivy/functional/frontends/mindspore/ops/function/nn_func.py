"""Includes Mindspore Frontend functions listed in the TODO list
https://github.com/unifyai/ivy/issues/14951."""

import ivy
import ivy.functional.frontends.paddle as paddle_frontend
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back

@with_supported_dtypes(
    {"2.5.0 and below": ("float32", "float16")},
    "Mindspore",
)

@to_ivy_arrays_and_back
def softmax(x, axis=-1):
    return ivy.softmax(x, axis=axis)
