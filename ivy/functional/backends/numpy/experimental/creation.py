# global
from typing import Optional, Tuple, Sequence, Union

import numpy as np

# local
from ivy.functional.backends.numpy.device import _to_device
import ivy

# Array API Standard #
# -------------------#


def vorbis_window(
    window_length: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    result = []
    for i in range(1, window_length * 2, 2):
        temp = np.sin(ivy.pi / 2 * (np.sin(ivy.pi * i / (window_length * 2)) ** 2))
        result.append(round(temp, 8))
    return np.array(result, dtype=dtype)


vorbis_window.support_native_out = False


def tril_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: str,
) -> Tuple[np.ndarray, ...]:
    return tuple(
        _to_device(np.asarray(np.tril_indices(n=n_rows, k=k, m=n_cols)), device=device)
    )


def hann_window(
    size: int,
    /,
    *,
    periodic: bool = True,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if size < 2:
        return np.ones([size], dtype=dtype)
    if periodic:
        count = np.arange(size) / size
    else:
        count = np.linspace(start=0, stop=size, num=size)
    return (0.5 - 0.5 * np.cos(2 * np.pi * count)).astype(dtype)


hann_window.support_native_out = False


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if window_length < 2:
        return np.ones([window_length], dtype=dtype)
    if periodic is False:
        return np.kaiser(M=window_length, beta=beta).astype(dtype)
    else:
        return np.kaiser(M=window_length + 1, beta=beta)[:-1].astype(dtype)


kaiser_window.support_native_out = False


def indices(
    dimensions: Sequence,
    dtype: np.dtype = np.int64,
    sparse: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    return np.indices(dimensions, dtype=dtype, sparse=sparse)


def unsorted_segment_min(
    data: np.ndarray,
    segment_ids: np.ndarray,
    num_segments: int,
) -> np.ndarray:
    ivy.utils.assertions.check_unsorted_segment_min_valid_params(
        data, segment_ids, num_segments
    )

    if data.dtype in [np.float32, np.float64]:
        init_val = np.finfo(data.dtype).max
    elif data.dtype in [np.int32, np.int64, np.int8, np.int16, np.uint8]:
        init_val = np.iinfo(data.dtype).max
    else:
        raise ValueError("Unsupported data type")

    res = np.full((num_segments,) + data.shape[1:], init_val, dtype=data.dtype)

    for i in range(num_segments):
        mask_index = segment_ids == i
        if np.any(mask_index):
            res[i] = np.min(data[mask_index], axis=0)

    return res


def unsorted_segment_sum(
    data: np.ndarray,
    segment_ids: np.ndarray,
    num_segments: int,
) -> np.ndarray:
    # Used the same check which is used for unsorted_segment_min as the
    # check should be same
    # Might require to change the assertion function name to
    # check_unsorted_segment_valid_params
    ivy.utils.assertions.check_unsorted_segment_min_valid_params(
        data, segment_ids, num_segments
    )

    res = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)

    for i in range(num_segments):
        mask_index = segment_ids == i
        if np.any(mask_index):
            res[i] = np.sum(data[mask_index], axis=0)

    return res


def mel_weight_matrix(
    num_mel_bins: int,
    dft_length: int,
    sample_rate: int,
    lower_edge_hertz: float = 0.0,
    upper_edge_hertz: float = 3000.0,
):
    lower_edge_hertz = np.array(lower_edge_hertz)
    upper_edge_hertz = np.array(upper_edge_hertz)
    zero = np.array(0.0)
    hz_to_mel = lambda f: 2595 * np.log10(1 + f / 700)
    nyquist_hz = sample_rate / 2
    linear_freqs = np.linspace(0, nyquist_hz, dft_length, dtype=np.float32)[1:]
    spec_bin_mels = hz_to_mel(linear_freqs)[..., None]
    mel_edges = np.linspace(
        hz_to_mel(lower_edge_hertz),
        hz_to_mel(upper_edge_hertz),
        num_mel_bins + 2,
        dtype=np.float32,
    )
    mel_edges = np.stack([mel_edges[i : i + 3] for i in range(num_mel_bins)])
    lower_edge_mel, center_mel, upper_edge_mel = [
        t.reshape((1, num_mel_bins)) for t in np.split(mel_edges, 3, axis=1)
    ]
    lower_slopes = (spec_bin_mels - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spec_bin_mels) / (upper_edge_mel - center_mel)
    mel_weights = np.maximum(zero, np.minimum(lower_slopes, upper_slopes))
    return np.pad(mel_weights, [[1, 0], [0, 0]])
