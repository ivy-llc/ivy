import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


_SWAP_DIRECTION_MAP = {
    None: "forward",
    "backward": "forward",
    "ortho": "ortho",
    "forward": "backward",
}


def _swap_direction(norm):
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        raise ValueError(
            f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".'
        ) from None


@to_ivy_arrays_and_back
def ifft(a, n=None, axis=-1, norm=None):
    a = ivy.array(a, dtype=ivy.complex128)
    if norm is None:
        norm = "backward"
    return ivy.ifft(a, axis, norm=norm, n=n)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.25.0 and below": ("float16",)}, "numpy")
def ifftshift(x, axes=None):
    x = ivy.asarray(x)

    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(
        axes,
        (int, type(ivy.uint8), type(ivy.uint16), type(ivy.uint32), type(ivy.uint64)),
    ):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    roll = ivy.roll(x, shift, axis=axes)

    return roll


@to_ivy_arrays_and_back
def fft(a, n=None, axis=-1, norm=None):
    return ivy.fft(ivy.astype(a, ivy.complex128), axis, norm=norm, n=n)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.25.0 and below": ("float16",)}, "numpy")
def fftshift(x, axes=None):
    x = ivy.asarray(x)

    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [(dim // 2) for dim in x.shape]
    elif isinstance(
        axes,
        (int, type(ivy.uint8), type(ivy.uint16), type(ivy.uint32), type(ivy.uint64)),
    ):
        shift = x.shape[axes] // 2
    else:
        shift = [(x.shape[ax] // 2) for ax in axes]

    roll = ivy.roll(x, shift, axis=axes)

    return roll


@with_unsupported_dtypes({"1.25.0 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def rfft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    a = ivy.array(a, dtype=ivy.float64)
    return ivy.dft(a, axis=axis, inverse=False, onesided=True, dft_length=n, norm=norm)


@with_unsupported_dtypes({"1.25.0 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def ihfft(a, n=None, axis=-1, norm=None):
    if n is None:
        n = a.shape[axis]
    norm = _swap_direction(norm)
    output = ivy.conj(rfft(a, n, axis, norm=norm).ivy_array)
    return output


@with_unsupported_dtypes({"1.25.0 and below": ("int",)}, "numpy")
@to_ivy_arrays_and_back
def fftfreq(n, d=1.0):
    if not isinstance(
        n, (int, type(ivy.int8), type(ivy.int16), type(ivy.int32), type(ivy.int64))
    ):
        raise ValueError("n should be an integer")

    N = (n - 1) // 2 + 1
    val = 1.0 / (n * d)
    results = ivy.empty(tuple([n]), dtype=int)

    p1 = ivy.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = ivy.arange(-(n // 2), 0, dtype=int)
    results[N:] = p2

    return results * val


@to_ivy_arrays_and_back
def rfftfreq(n, d=1.0):
    if not isinstance(
        n, (int, type(ivy.int8), type(ivy.int16), type(ivy.int32), type(ivy.int64))
    ):
        raise ValueError("n should be an integer")

    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = ivy.arange(0, N, dtype=int)
    return results * val


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.24.3 and above": ("float16",)}, "numpy")
def rfft2(a, s=None, axes=(-2, -1), norm=None):
    if len(axes) != 2:
        raise ValueError("rfft2 only supports 2 axes")

    if s is not None and any(dim < 0 for dim in s):
        raise ValueError("Shape s must be non-negative")

    if s is not None and len(s) != len(axes):
        raise ValueError("Shape s and axes must have the same length")

    if len(set(axes)) != len(axes):
        raise ValueError("Repeated axes are not allowed")

    if len(axes) > 3:
        raise ValueError("rfft2 does not support more than 3 axes")
    if s:
        cropped_shape = tuple(min(dim, size) for dim, size in zip(a.shape, s))
        pad_width = tuple((0, size - dim) for dim, size in zip(cropped_shape, s))
        a = ivy.pad(a, pad_width)

    s = s or tuple(a.shape[axis] for axis in axes)

    a = ivy.astype(a, "float64")
    norm = norm or "backward"

    for axis in axes:
        a = ivy.dft(a, axes=axis, inverse=False, onesided=True, dft_length=s, norm=norm)

    return a
