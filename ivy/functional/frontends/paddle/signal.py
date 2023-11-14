import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


# istft
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
def istft(
    stft_matrix,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    normalized=False,
    onesided=True,
    length=None,
    name=None,
):
    stft_matrix = ivy.asarray(stft_matrix)
    return ivy.istft(
        stft_matrix,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        length=length,
        name=name,
    )
