# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def fft2(x, n=None, axes=(- 2, - 1), norm="backward", name=None):
    return ivy.fft2(x, axes, norm=norm, n=n)



