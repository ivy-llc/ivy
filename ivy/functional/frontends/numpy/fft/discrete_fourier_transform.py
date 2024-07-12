import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes


_SWAP_DIRECTION_MAP = {
    None: "forward",
    "backward": "forward",
    "ortho": "ortho",
    "forward": "backward",
}


# --- Helpers --- #
# --------------- #


def _swap_direction(norm):
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        raise ValueError(
            f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".'
        ) from None


# --- Main --- #
# ------------ #


@to_ivy_arrays_and_back
def fft(a, n=None, axis=-1, norm=None):
    return ivy.fft(ivy.astype(a, ivy.complex128), axis, norm=norm, n=n)


@with_unsupported_dtypes({"1.26.3 and below": ("int",)}, "numpy")
@to_ivy_arrays_and_back
def fftfreq(n, d=1.0):
    if not isinstance(
        n, (int, type(ivy.int8), type(ivy.int16), type(ivy.int32), type(ivy.int64))
    ):
        raise TypeError("n should be an integer")

    N = (n - 1) // 2 + 1
    val = 1.0 / (n * d)
    results = ivy.empty((n,), dtype=int)

    p1 = ivy.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = ivy.arange(-(n // 2), 0, dtype=int)
    results[N:] = p2

    return results * val


@with_supported_dtypes(
    {"1.26.0 and below": ("float32", "float64", "complex64", "complex128")},
    "numpy",
)
@to_ivy_arrays_and_back
def fftn(a, s=None, axes=None, norm=None):
    invreal = 0
    if norm is None:
        norm = "backward"
    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            axes = [ax % len(a.shape) for ax in axes]
            s = ivy.gather(a.shape, ivy.array(axes, dtype="int64"))
    else:
        shapeless = 0
    s = list(s)
    if axes is None:
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    itl = list(range(len(axes)))
    itl.reverse()
    for ii in itl:
        a = ivy.fft(a, axes[ii], norm=norm, n=int(s[ii]))
    return ivy.astype(a, ivy.complex128)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.26.3 and below": ("float16",)}, "numpy")
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


@to_ivy_arrays_and_back
def ifft(a, n=None, axis=-1, norm=None):
    a = ivy.array(a, dtype=ivy.complex128)
    if norm is None:
        norm = "backward"
    return ivy.ifft(a, axis, norm=norm, n=n)


@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def ifft2(a, s=None, axes=(-2, -1), norm=None):
    a = ivy.asarray(a, dtype=ivy.complex128)
    a = ivy.ifftn(a, s=s, axes=axes, norm=norm)
    return a


@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def ifftn(a, s=None, axes=None, norm=None):
    a = ivy.asarray(a, dtype=ivy.complex128)
    a = ivy.ifftn(a, s=s, axes=axes, norm=norm)
    return a


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.26.3 and below": ("float16",)}, "numpy")
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


@with_unsupported_dtypes({"1.26.3 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def ihfft(a, n=None, axis=-1, norm=None):
    if n is None:
        n = a.shape[axis]
    norm = _swap_direction(norm)
    output = ivy.conj(rfft(a, n, axis, norm=norm).ivy_array)
    return output


@with_unsupported_dtypes({"1.26.3 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def rfft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    a = ivy.array(a, dtype=ivy.float64)
    return ivy.dft(a, axis=axis, inverse=False, onesided=True, dft_length=n, norm=norm)


@to_ivy_arrays_and_back
def rfftfreq(n, d=1.0):
    if not isinstance(
        n, (int, type(ivy.int8), type(ivy.int16), type(ivy.int32), type(ivy.int64))
    ):
        raise TypeError("n should be an integer")

    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = ivy.arange(0, N, dtype=int)
    return results * val


@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def rfftn(a, s=None, axes=None, norm=None):
    a = ivy.asarray(a, dtype=ivy.complex128)
    return ivy.rfftn(a, s=s, axes=axes, norm=norm)
