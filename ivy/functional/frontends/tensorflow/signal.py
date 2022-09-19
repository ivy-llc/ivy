
import numpy as np
#hamming_window

#hann_window
def hann_window(length, periodic=True):
    """Returns a 1D Hann window.

    The Hann window is a taper formed by using a weighted cosine.

    Args:
        length: An integer, the number of points in the returned window.
        periodic: A boolean, controls the periodicity of the returned window.
            If True (default), the returned window is periodic with value 1 at
            the end-points. If False, the returned window is symmetric.

    Returns:
        A 1D array containing the Hann window.

    Raises:
        ValueError: If `length` is not positive.
    """
    if length < 1:
        raise ValueError("Window length must be a positive integer.")
    if length == 1:
        return np.ones(1, dtype=np.float32)
    odd = length % 2
    if not periodic:
        length += 1
    n = np.arange(0, length)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
    if not periodic:
        w = w[:-1]
    return w

#inverse_stft
def inverse_stft(stfts, frame_step, frame_length=None, fft_length=None,
                 window_fn=hann_window, pad_end=False):
    """Computes the inverse of Short-time Fourier Transform (STFT).

    Args:
        stfts: A complex64 `Tensor` of shape [batch_size, ?, fft_unique_bins]
            (or [?, fft_unique_bins]) where `?` represents any dimension.
        frame_step: An integer, the number of samples to step.
        frame_length: An integer, the length of the FFT window. Defaults to
            `frame_step`.
        fft_length: An integer, size of the FFT to apply. If `None`, the full
            signal is used (padded with zeros for odd-length signals). Defaults
            to `frame_length`.
        window_fn: A function that takes a 1D integer tensor and returns a 1D
            tensor of the same type and shape as the input. Defaults to
            `hann_window`.
        pad_end: A boolean, whether to pad the end of the signal with zeros to
            ensure that all frames are full. Defaults to `False`.

    Returns:
        A float32 `Tensor` of shape [batch_size, ?] (or [?]) where `?` represents
        the same dimension as the input `stfts`.

    Raises:
        ValueError: If `stfts` is not a 2D or 3D `Tensor`, or if `frame_step` is
            not positive, or if `frame_length` is not positive, or if `fft_length`
            is not positive, or if `window_fn` is not callable, or if `pad_end` is
            not a boolean.
    """
    if not isinstance(stfts, tf.Tensor):
        raise TypeError("stfts must be a Tensor")
    if len(stfts.get_shape()) not in [2, 3]:
        raise ValueError("stfts must be a 2D or 3D Tensor")
    if not isinstance(frame_step, int) or frame_step < 1:
        raise ValueError("frame_step must be a positive integer")
    if frame_length is None:
        frame_length = frame_step
    if not isinstance(frame_length, int) or frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if fft_length is None:
        fft_length = frame_length
    if not isinstance(fft_length, int) or fft_length < 1:
        raise ValueError("fft_length must be a positive integer")
    if not callable(window_fn):
        raise ValueError("window_fn must be callable")
    if not isinstance(pad_end, bool):
        raise ValueError("pad_end must be a boolean")
        