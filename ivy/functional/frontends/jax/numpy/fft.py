# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"0.4.8 and below": ("bool",)}, "jax")
def fft(a, n=None, dim=-1, norm=None):
    return ivy.fft(a, dim, n=n, norm=norm)
