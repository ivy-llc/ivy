# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
def fft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    return ivy.fft(a, axis, norm=norm, n=n)


@to_ivy_arrays_and_back
def fft2(a, s=None, axes=(-2, -1), norm=None):
    if norm is None:
        norm = "backward"
    return ivy.array(ivy.fft2(a, s=s, dim=axes, norm=norm), dtype=ivy.dtype(a))


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
def ifft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    return ivy.ifft(a, axis, norm=norm, n=n)


@to_ivy_arrays_and_back
def ifft2(a, s=None, axes=(-2, -1), norm=None):
    if norm is None:
        norm = "backward"
    return ivy.array(ivy.ifft2(a, s=s, dim=axes, norm=norm), dtype=ivy.dtype(a))


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def rfft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    if n is None:
        n = len(a)
    if ivy.current_backend_str() == "tensorflow":
        if a.dtype in ["uint64", "int64", "float64"]:
            a_new = ivy.astype(a, "complex128")
        else:
            a_new = ivy.astype(a, "complex64")
    else:
        a_new = a
    fft_fun = ivy.fft
    return fft_fun(a_new, axis, norm=norm, n=n)[: n // 2 + 1]
