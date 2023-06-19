# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.4.2 and below": ("complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def fft(x, n=None, axis=-1.0, norm="backward", name=None):
    ret = ivy.fft(ivy.astype(x, "complex128"), axis, norm=norm, n=n)
    return ivy.astype(ret, x.dtype)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def rfft(x, n=None, axis=-1.0, norm="backward", name=None):
    ret = ivy.rfft(ivy.astype(x, "float64"), axis, norm=norm, n=n)
    return ret
