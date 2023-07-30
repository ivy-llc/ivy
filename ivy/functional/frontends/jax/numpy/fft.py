# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
import ivy.functional.frontends.jax as jax_frontend


if ivy.backend in ("jax", "numpy", "tensorflow"):
    double_precision_dtypes = (ivy.complex128, ivy.float64, ivy.int64, ivy.uint64)
else:
    double_precision_dtypes = (ivy.complex128, ivy.float64, ivy.int64)


@to_ivy_arrays_and_back
def ifft(a, n=None, axis=-1, norm=None):
    # JAX applies no normalization if norm=None, which is
    # equivalent to norm='backward' in the ivy implementation
    if norm is None:
        norm = "backward"

    # Make sure return dtype precision matches
    # ground truth JAX return dtype precision
    if jax_frontend.config.jax_enable_x64:
        a = ivy.array(a)
        input_dtype = a.dtype
        if input_dtype in double_precision_dtypes:
            a = a.astype(ivy.complex128)
            return ivy.ifft(a, axis, norm=norm, n=n)
    a = ivy.array(a, dtype=ivy.complex64)
    return ivy.ifft(a, axis, norm=norm, n=n)


@to_ivy_arrays_and_back
def fft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    return ivy.fft(a, axis, norm=norm, n=n)
