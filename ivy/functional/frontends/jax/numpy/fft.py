# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def fft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    return ivy.fft(a, axis, norm=norm, n=n)
