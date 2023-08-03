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
