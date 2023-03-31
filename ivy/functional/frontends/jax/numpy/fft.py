# local
import ivy
from ivy.functional.frontends.jax import DeviceArray
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def fft(a, n=None, axis=-1, norm=None):
    return ivy.fft(a, axis)
