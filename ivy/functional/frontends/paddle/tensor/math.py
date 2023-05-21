# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def sin(x, name=None):
    return ivy.sin(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def cos(x, name=None):
    return ivy.cos(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def acos(x, name=None):
    return ivy.acos(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def cosh(x, name=None):
    return ivy.cosh(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def tanh(x, name=None):
    return ivy.tanh(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def acosh(x, name=None):
    return ivy.acosh(x)
