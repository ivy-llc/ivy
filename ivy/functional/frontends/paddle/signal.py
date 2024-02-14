import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import with_supported_dtypes


# stft function
@with_supported_dtypes(
    {
        "2.5.1 and below": (
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def stft(
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
    x = ivy.asarray(x)
    return ivy.stft(
        x,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        name=name,
    )
