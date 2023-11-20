import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
)
from ivy.func_wrapper import with_supported_dtypes


# dct
@to_ivy_arrays_and_back
def dct(input, type=2, n=None, axis=-1, norm=None, name=None):
    return ivy.dct(input, type=type, n=n, axis=axis, norm=norm)


# idct
@to_ivy_arrays_and_back
def idct(input, type=2, n=None, axis=-1, norm=None, name=None):
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return ivy.dct(input, type=inverse_type, n=n, axis=axis, norm=norm)


# kaiser_bessel_derived_window
@handle_tf_dtype
@to_ivy_arrays_and_back
def kaiser_bessel_derived_window(
    window_length, beta=12.0, dtype=ivy.float32, name=None
):
    return ivy.kaiser_bessel_derived_window(window_length, beta=beta, dtype=dtype)


@with_supported_dtypes(
    {"2.15.0 and below": ("float32", "float64", "float16", "bfloat16")},
    "tensorflow",
)
@handle_tf_dtype
@to_ivy_arrays_and_back
def kaiser_window(window_length, beta=12.0, dtype=ivy.float32, name=None):
    return ivy.kaiser_window(window_length, periodic=False, beta=beta, dtype=dtype)


# stft
@to_ivy_arrays_and_back
def stft(
    signals,
    frame_length,
    frame_step,
    fft_length=None,
    window_fn=None,
    pad_end=False,
    name=None,
):
    signals = ivy.asarray(signals)
    return ivy.stft(
        signals,
        frame_length,
        frame_step,
        fft_length=fft_length,
        window_fn=window_fn,
        pad_end=pad_end,
        name=name,
    )


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64", "bfloat16")},
    "tensorflow",
)
@to_ivy_arrays_and_back
def vorbis_window(window_length, dtype=ivy.float32, name=None):
    return ivy.vorbis_window(window_length, dtype=dtype, out=None)


kaiser_bessel_derived_window.supported_dtypes = (
    "float32",
    "float64",
    "float16",
    "bfloat16",
)
