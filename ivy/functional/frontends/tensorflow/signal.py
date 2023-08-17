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


# dct
@to_ivy_arrays_and_back
def dct(input, type=2, n=None, axis=-1, norm=None, name=None):
    return ivy.dct(input, type=type, n=n, axis=axis, norm=norm)


# vorbis_window
@to_ivy_arrays_and_back
def vorbis_window(window_length, dtype=ivy.float32, name=None):
    return ivy.vorbis_window(window_length, dtype=dtype, out=None)


vorbis_window.supported_dtypes = ("float32", "float64", "float16", "bfloat16")


# idct
@to_ivy_arrays_and_back
def idct(input, type=2, n=None, axis=-1, norm=None, name=None):
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return ivy.dct(input, type=inverse_type, n=n, axis=axis, norm=norm)

# stft
@to_ivy_arrays_and_back
def stft(signals, frame_length, frame_step, fft_length=None,
         window_fn=ivy.hann_window, pad_end=False, name=None):

    def stft_1D(signals, frame_length, frame_step, fft_length, pad_end):

        num_samples = signals.shape[-1]

        if pad_end:
            num_samples = signals.shape[-1]
            num_frames = -(-num_samples // frame_step)
            pad_length = max(0, frame_length + frame_step * (num_frames - 1) - num_samples)

            signals = ivy.pad(signals, [(0, pad_length)])
        else:
            num_frames = 1 + (num_samples - frame_length) // frame_step

        stft_result = []

        window = window_fn(frame_length, dtype=ivy.float32)

        for i in range(num_frames):

            start = i * frame_step
            end = start + frame_length
            frame = signals[..., start:end]
            windowed_frame = frame * window
            pad_length = fft_length - frame_length
            windowed_frame = ivy.pad(windowed_frame, [(0, pad_length)])

            fft_frame = ivy.fft(ivy.astype(windowed_frame, ivy.complex64), -1)
            slit = int((fft_length // 2 + 1))
            stft_result.append(fft_frame[..., 0:slit])

        stft = ivy.stack(stft_result, axis=0)
        return ivy.asarray(stft)

    def stft_helper(nested_list, frame_length, frame_step, fft_length):

        nested_list = ivy.asarray(nested_list)
        if len(ivy.shape(nested_list)) > 1:
            return [stft_helper(sublist, frame_length, frame_step, fft_length)
                    for sublist in nested_list]
        else:
            return stft_1D(nested_list, frame_length, frame_step, fft_length, pad_end)

    return ivy.asarray(stft_helper(signals, frame_length, frame_step, fft_length))