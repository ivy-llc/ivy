# global
import ivy
from ivy.functional.frontends.scipy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def fft(x, n=None, axis=-1, norm="backward", overwrite_x=False):
    return ivy.fft(x, axis, norm=norm, n=n)


@to_ivy_arrays_and_back
def ifft(x, n=None, axis=-1, norm="backward", overwrite_x=False):
    return ivy.ifft(x, axis, norm=norm, n=n)
