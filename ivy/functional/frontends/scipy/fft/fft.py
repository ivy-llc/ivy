# global
import ivy
from ivy.functional.frontends.scipy.func_wrapper import (
    to_ivy_arrays_and_back,
)


# fft
@to_ivy_arrays_and_back
def fft(x, n=None, axis=-1, norm="backward", overwrite_x=False):
    return ivy.fft(x, axis, norm=norm, n=n)


# ifft
@to_ivy_arrays_and_back
def ifft(x, n=None, axis=-1, norm="backward", overwrite_x=False):
    return ivy.ifft(x, axis, norm=norm, n=n)


# dct
@to_ivy_arrays_and_back
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, orthogonalize=None):
    return ivy.dct(x, type=type, n=n, axis=axis, norm=norm)


# idct
@to_ivy_arrays_and_back
def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, orthogonalize=None):
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return ivy.dct(x, type=inverse_type, n=n, axis=axis, norm=norm)


@to_ivy_arrays_and_back
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False):
    return ivy.fft2(x, s=s, dim=axes, norm=norm)


@to_ivy_arrays_and_back
def ifftn(
    x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    return ivy.ifftn(x, s=s, dim=axes, norm=norm)


@to_ivy_arrays_and_back
def rfftn(
    x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    return ivy.rfftn(x, s=s, dim=axes, norm=norm)
