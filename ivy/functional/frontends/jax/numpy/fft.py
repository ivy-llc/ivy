# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
import ivy.functional.frontends.jax as jax_frontend
from ivy.func_wrapper import with_unsupported_dtypes


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
@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
def fftshift(x, axes=None, name=None):
    shape = x.shape

    if axes is None:
        axes = tuple(range(x.ndim))
        shifts = [(dim // 2) for dim in shape]
    elif isinstance(axes, int):
        shifts = shape[axes] // 2
    else:
        shifts = [shape[ax] // 2 for ax in axes]

    roll = ivy.roll(x, shifts, axis=axes)

    return roll

@to_ivy_arrays_and_back
def fft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    return ivy.fft(a, axis, norm=norm, n=n)
