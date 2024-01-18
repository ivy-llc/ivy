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
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
def fftfreq(n, d=1.0, *, dtype=None):
    if not isinstance(
        n, (int, type(ivy.int8), type(ivy.int16), type(ivy.int32), type(ivy.int64))
    ):
        raise TypeError("n should be an integer")

    dtype = ivy.float64 if dtype is None else ivy.as_ivy_dtype(dtype)

    N = (n - 1) // 2 + 1
    val = 1.0 / (n * d)

    results = ivy.zeros((n,), dtype=dtype)
    results[:N] = ivy.arange(0, N, dtype=dtype)
    results[N:] = ivy.arange(-(n // 2), 0, dtype=dtype)

    return results * val


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


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.25.2 and below": ("float16", "bfloat16")}, "numpy")
def rfft(a, n=None, axis=-1, norm=None):
    if n is None:
        n = a.shape[axis]
    if norm is None:
        norm = "backward"
    result = ivy.dft(
        a, axis=axis, inverse=False, onesided=False, dft_length=n, norm=norm
    )
    slices = [slice(0, a) for a in result.shape]
    slices[axis] = slice(0, int(ivy.shape(result, as_array=True)[axis] // 2 + 1))
    result = result[tuple(slices)]
    return result
