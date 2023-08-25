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


# stft
@to_ivy_arrays_and_back
def stft(signals, frame_length, frame_step, fft_length=None,
        window_fn=ivy.hann_window, pad_end=False, name=None):

    if not isinstance(frame_length, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(frame_length)}"
        )

    if not isinstance(frame_step, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(frame_step)}"
        )

    if fft_length is not None:
        if not isinstance(fft_length, int):
            raise ivy.utils.exceptions.IvyError(
                f"Expecting <class 'int'> instead of {type(fft_length)}"
            )

    input_dtype = ivy.dtype(signals)
    if input_dtype == ivy.float32:
        dtype = ivy.complex64
    elif input_dtype == ivy.float64:
        dtype = ivy.complex128

    def stft_1D(signals, frame_length, frame_step, fft_length, pad_end):

        if fft_length == None:
            fft_length = 1
            while fft_length < frame_length:
                fft_length *= 2

        num_samples = signals.shape[-1]

        if pad_end:
            num_samples = signals.shape[-1]
            num_frames = -(-num_samples // frame_step)
            pad_length = max(0, frame_length + frame_step * (num_frames - 1) - num_samples)

            signals = ivy.pad(signals, [(0, pad_length)])
        else:
            num_frames = 1 + (num_samples - frame_length) // frame_step

        stft_result = []

        if window_fn == None:
            window = 1
        else:
            window = window_fn(frame_length)

        for i in range(num_frames):

            start = i * frame_step
            end = start + frame_length
            frame = signals[..., start:end]
            windowed_frame = frame * window
            pad_length = fft_length - frame_length
            windowed_frame = ivy.pad(windowed_frame, [(0, pad_length)])
            windowed_frame = ivy.astype(windowed_frame, dtype)

            fft_frame = ivy.fft(windowed_frame, -1)
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

    to_return = ivy.asarray(stft_helper(signals, frame_length, frame_step, fft_length))
    return ivy.astype(to_return, dtype)
