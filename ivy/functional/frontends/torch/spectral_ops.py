import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def bartlett_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False
):
    # this implementation is based on scipy.signal.windows.bartlett
    # https://github.com/scipy/scipy/blob/v1.11.2/scipy/signal/windows/_windows.py#L625-L721
    if int(window_length) != window_length or window_length < 0:
        raise ValueError("Window length must be a non-negative integer")
    elif window_length == 1:
        return ivy.ones(window_length)
    else:
        N = window_length + 1 if periodic else window_length

        res = ivy.arange(0, N, dtype=dtype)
        res = ivy.where(
            ivy.less_equal(res, (N - 1) / 2.0),
            2.0 * res / (N - 1),
            2.0 - 2.0 * res / (N - 1),
        )

        return res[:-1] if periodic else res


@to_ivy_arrays_and_back
def stft(
    input,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    pad_mode="reflect",
    normalized=False,
    onesided=True,
    return_complex=None,
):
    input = ivy.asarray(input)
    assert len(input.shape) == 1 or len(input.shape) == 2

    if window is not None:
        assert hop_length is not None and win_length is not None

    if not hop_length:
        hop_length = n_fft // 4

    if not win_length:
        if not window:
            win_length = n_fft
        else:
            win_length = len(window)

    if window is None:
        window = ivy.ones((win_length))

    if win_length < n_fft:
        len_diff = n_fft - win_length
        padding = ((len_diff) // 2, (len_diff + 1) // 2)
        window = ivy.pad(window, padding, mode="constant", constant_values=0)
        win_length = n_fft

    window_function = lambda x: ivy.multiply(x, window)

    if center:
        padding = (n_fft // 2, n_fft // 2)
        if len(input.shape) == 2:
            padding = ((0, 0), (n_fft // 2, n_fft // 2))
        input = ivy.pad(input, padding, mode=pad_mode)

    result = (
        ivy.stft(
            input, win_length, hop_length, fft_length=n_fft, window_fn=window_function
        )
        / n_fft
    )

    if normalized:
        result *= (n_fft) ** (-0.5)

    transposed_results = ivy.matrix_transpose(result)

    if not onesided:
        if len(input.shape) == 1:
            transposed_results = ivy.vstack(
                [transposed_results, transposed_results[: n_fft // 2 - (1 - n_fft % 2)]]
            )
        elif len(input.shape) == 2:
            result_T = [
                ivy.vstack([i, i[: (n_fft // 2 - 1 + n_fft % 2)]])
                for i in transposed_results
            ]
            transposed_results = ivy.stack(result_T)

    if return_complex:
        return transposed_results

    else:
        real = ivy.real(transposed_results)
        imag = ivy.imag(transposed_results)
        return ivy.stack([real, imag], axis=-1)
