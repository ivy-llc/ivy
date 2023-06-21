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
@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
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
@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
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


@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def rfft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    a = ivy.array(a, dtype=ivy.float64)
    return ivy.dft(a, axis=axis, inverse=False, onesided=True, dft_length=n, norm=norm)


@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def ihfft(a, n=None, axis=-1, norm=None):
    if n is None:
        n = a.shape[axis]
    norm = _swap_direction(norm)
    output = ivy.conj(rfft(a, n, axis, norm=norm).ivy_array)
    return output


@with_unsupported_dtypes({"1.24.3 and below": ("int",)}, "numpy")
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


@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def ifftn(a, s=None, axes=None, norm=None):
    a = ivy.array(a, dtype=ivy.complex128)
    if s is not None:
        if not isinstance(s, tuple):
            raise TypeError("'int' object is not iterable")

    if axes is not None:
        if not isinstance(axes, tuple):
            raise TypeError("'int' object is not iterable")

    if axes is None and s is None:
        axes = tuple(range(a.ndim))

    elif axes is None and s is not None:
        if len(s) > len(a.shape):
            raise ValueError(
                f"axis -{len(a.shape) + 1} is out of bounds for array of dimension"
                f" {len(a.shape)}"
            )
        else:
            last_axes = []
            num_axes = len(a.shape)
            num_last_axes = len(s)

            for i in range(num_last_axes):
                last_axes.append(num_axes + i - num_last_axes)

            axes = tuple(last_axes)

    elif axes is not None and s is not None:
        if len(s) != len(axes):
            raise ValueError("Shape and axes have different lengths.")

    if s is None:
        output_shape = []
        for axis in axes:
            output_shape.append(a.shape[axis])
        s = tuple(output_shape)

    if norm is None:
        norm = "backward"

    for i in reversed(range(len(axes))):
        a = ivy.ifft(a, dim=axes[i], norm=norm, n=s[i])

    return a

