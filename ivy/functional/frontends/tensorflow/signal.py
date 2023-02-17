import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
)


@handle_tf_dtype
@to_ivy_arrays_and_back
def kaiser_window(window_length, beta=12.0, dtype=ivy.float32, name=None):
    return ivy.kaiser_window(window_length, periodic=False, beta=beta, dtype=dtype)


kaiser_window.supported_dtypes = ("float32", "float64", "float16", "bfloat16")


@handle_tf_dtype
@to_ivy_arrays_and_back
def stft(signal, frame_length, frame_step, fft_length=None,
         window_fn="hann_window", pad_end=False, name=None):
    return ivy.stft(signal, frame_step=frame_step, frame_length=frame_length,
                    fft_length=fft_length, window_fn=window_fn,
                    pad_end=pad_end, name=name)
