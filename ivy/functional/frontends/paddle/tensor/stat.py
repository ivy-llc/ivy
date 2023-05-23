# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def mean(input, axis=None, keepdims=False, out=None):
    return ivy.std(input, axis=axis, keepdims=keepdims, out=None)
