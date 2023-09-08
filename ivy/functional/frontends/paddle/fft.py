# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.5.1 and below": ("complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def fft(x, n=None, axis=-1.0, norm="backward", name=None):
    ret = ivy.fft(ivy.astype(x, "complex128"), axis, norm=norm, n=n)
    return ivy.astype(ret, x.dtype)


@with_supported_dtypes(
    {
        "2.5.1 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def fftshift(x, axes=None, name=None):
    shape = x.shape

    if axes is None:
        axes = tuple(range(x.ndim))
        shifts = [(dim // 2) for dim in shape]
    elif isinstance(axes, int):
        shifts = shape[axes] // 2
    else:
        shifts = ivy.concat([shape[ax] // 2 for ax in axes])

    roll = ivy.roll(x, shifts, axis=axes)

    return roll


@with_supported_dtypes(
    {"2.5.1 and below": ("complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def hfft(x, n=None, axis=-1, norm="backward", name=None):
    """Compute the FFT of a signal that has Hermitian symmetry, resulting in a real
    spectrum."""
    # Determine the input shape and axis length
    input_shape = x.shape
    input_len = input_shape[axis]

    # Calculate n if not provided
    if n is None:
        n = 2 * (input_len - 1)

    # Perform the FFT along the specified axis
    result = ivy.fft(x, axis, n=n, norm=norm)

    return ivy.real(result)


@with_supported_dtypes(
    {"2.5.1 and below": ("complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def ifft(x, n=None, axis=-1.0, norm="backward", name=None):
    ret = ivy.ifft(ivy.astype(x, "complex128"), axis, norm=norm, n=n)
    return ivy.astype(ret, x.dtype)


@with_supported_dtypes(
    {
        "2.5.1 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def ifftshift(x, axes=None, name=None):
    shape = x.shape

    if axes is None:
        axes = tuple(range(x.ndim))
        shifts = [-(dim // 2) for dim in shape]
    elif isinstance(axes, int):
        shifts = -(shape[axes] // 2)
    else:
        shifts = ivy.concat([-shape[ax] // 2 for ax in axes])

    roll = ivy.roll(x, shifts, axis=axes)

    return roll


@with_supported_dtypes(
    {"2.5.1 and below": ("complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def irfft(x, n=None, axis=-1.0, norm="backward", name=None):
    if n is None:
        n = 2 * (x.shape[axis] - 1)

    pos_freq_terms = ivy.take_along_axis(x, range(n // 2 + 1), axis)
    neg_freq_terms = ivy.conj(pos_freq_terms[1:-1][::-1])
    combined_freq_terms = ivy.concat((pos_freq_terms, neg_freq_terms), axis=axis)
    time_domain = ivy.ifft(combined_freq_terms, axis, norm=norm, n=n)
    if ivy.isreal(x):
        time_domain = ivy.real(time_domain)
    return time_domain


@with_supported_dtypes(
    {
        "2.5.1 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def  stft(
    x,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    pad_mode="reflect",
    normalized=False,
    onesided=True,
    name=None,
):
    # Calculate the STFT
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = n_fft // 4

    if window is None:
        window = ivy.ones(n_fft)  # Default rectangular window

    # Ensure the input signal is Ivy array
    x = ivy.array(x)

    # Calculate the STFT
    x_len = x.shape[-1]
    num_frames = 1 + (x_len - n_fft) // hop_length
    stft_result = ivy.zeros((num_frames, n_fft), dtype=x.dtype)

    for t in range(num_frames):
        start = t * hop_length
        end = start + n_fft

        if center:
            pad_before = max(0, (win_length - n_fft) // 2)
            pad_after = max(0, win_length - n_fft - pad_before)
            padded_signal = ivy.pad(x[..., start:end], [(0, 0)] * (x.ndim - 1) + [(pad_before, pad_after)])
        else:
            padded_signal = x[..., start:end]

        frame = padded_signal * window
        frame_fft = ivy.fft(frame)
        stft_result[t] = frame_fft

    if onesided:
        stft_result = stft_result[..., :n_fft // 2 + 1]

    if normalized:
        stft_result /= ivy.sqrt(n_fft)

    return stft_result
