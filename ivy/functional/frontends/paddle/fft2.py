# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.5.0 and below": ("complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def fft2(x, s=None, axes=(- 2, - 1), norm='backward', name=None):
    ret = ivy.fft2(ivy.astype(x, "complex128"), axes, norm=norm, n=s)
    return ivy.astype(ret, x.dtype)