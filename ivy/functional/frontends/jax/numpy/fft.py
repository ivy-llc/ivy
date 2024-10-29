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


@with_unsupported_dtypes({"1.24.3 and below": ("complex64", "bfloat16")}, "numpy")
@to_ivy_arrays_and_back
def ifftn(a, s=None, axes=None, norm=None):
    a = ivy.asarray(a, dtype=ivy.complex128)
    a = ivy.ifftn(a, s=s, axes=axes, norm=norm)
    return a


@to_ivy_arrays_and_back
def ifftshift(x, axes=None):
    if not ivy.is_array(x):
        raise ValueError("Input 'x' must be an array")

    # Get the shape of x
    shape = ivy.shape(x)

    # If axes is None, shift all axes
    if axes is None:
        axes = tuple(range(x.ndim))

    # Convert axes to a list if it's not already
    axes = [axes] if isinstance(axes, int) else list(axes)

    # Perform the shift for each axis
    for axis in axes:
        axis_size = shape[axis]
        shift = -ivy.floor(axis_size / 2).astype(ivy.int32)
        result = ivy.roll(x, shift, axis=axis)

    return result


@to_ivy_arrays_and_back
def irfftn(a, s=None, axes=None, norm=None):
    x = ivy.asarray(a)

    if axes is None:
        axes = list(range(len(x.shape)))

    include_last_axis = len(x.shape) - 1 in axes

    if s is None:
        s = [
            x.shape[axis] if axis != (len(x.shape) - 1) else 2 * (x.shape[axis] - 1)
            for axis in axes
        ]

    real_result = x
    remaining_axes = [axis for axis in axes if axis != (len(x.shape) - 1)]

    if remaining_axes:
        real_result = ivy.ifftn(
            x,
            s=[s[axes.index(axis)] for axis in remaining_axes],
            axes=remaining_axes,
            norm=norm,
        )

    if include_last_axis:
        axis = len(x.shape) - 1
        size = s[axes.index(axis)]
        freq_domain = ivy.moveaxis(real_result, axis, -1)
        slices = [slice(None)] * ivy.get_num_dims(freq_domain)
        slices[-1] = slice(0, size // 2 + 1)
        pos_freq_terms = freq_domain[tuple(slices)]
        slices[-1] = slice(1, -1)
        neg_freq_terms = ivy.conj(pos_freq_terms[tuple(slices)][..., ::-1])
        combined_freq_terms = ivy.concat((pos_freq_terms, neg_freq_terms), axis=-1)
        real_result = ivy.ifftn(combined_freq_terms, s=[size], axes=[-1], norm=norm)
        real_result = ivy.moveaxis(real_result, -1, axis)

    if ivy.is_complex_dtype(x.dtype):
        output_dtype = ivy.float32 if x.dtype == ivy.complex64 else ivy.float64
    else:
        if str(x.dtype) in ["float64", "int64", "uint64"]:
            output_dtype = ivy.float64
        else:
            output_dtype = ivy.float32

    result_t = ivy.astype(real_result, output_dtype)
    return result_t


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
