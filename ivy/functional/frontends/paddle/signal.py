from paddle.signal import stft as paddle_stft


def short_DFT(
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
    """
    Compute the Short-Time Fourier Transform (STFT) using PaddlePaddle.

    Parameters
    ----------
    x : paddle.Tensor
        The input data, which can be 1D or 2D Tensor with shape [..., seq_length].
        It can be real-valued or complex Tensor.

    n_fft : int
        Number of input samples for Fourier transform.

    hop_length : int, optional
        Steps to advance between adjacent windows. Must be > 0.
        Default: None (treated as n_fft // 4).

    win_length : int, optional
        Size of the window. Default: None (equal to n_fft).

    window : paddle.Tensor, optional
        1D tensor of size win_length. Center-padded to length n_fft if win_length < n_fft.
        Default: None (rectangle window with value 1 of size win_length).

    center : bool, optional
        Pad x so that t × hop_length is at the center of t-th frame. Default: True.

    pad_mode : str, optional
        Padding pattern when center is True. See paddle.nn.functional.pad for options.
        Default: "reflect".

    normalized : bool, optional
        Scale the output by 1/sqrt(n_fft). Default: False.

    onesided : bool, optional
        Return half of the Fourier output that satisfies the conjugate symmetry condition
        when input is real-valued. Cannot be True if input is complex. Default: True.

    name : str, optional
        Default is None. Normally, no need for the user to set this. Refer to Name.

    Returns
    -------
    paddle.Tensor
        Complex STFT output tensor with shape [..., n_fft//2 + 1, num_frames] (real-valued input
        and onesided is True) or [..., n_fft, num_frames] (onesided is False).
    """
    return paddle_stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        name=name,
    )
