# global
from typing import Optional, Tuple

import numpy as np

# local
from ivy.functional.backends.numpy.device import _to_device
import ivy

# Array API Standard #
# -------------------#


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: str,
) -> Tuple[np.ndarray]:
    return tuple(
        _to_device(np.asarray(np.triu_indices(n=n_rows, k=k, m=n_cols)), device=device)
    )


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
    if size == 1:
        return np.array([1], dtype=dtype)
    if size == 2 and periodic is False:
        return np.array([0, 1], dtype=dtype)
    if periodic is False:
        return np.array(np.hanning(size), dtype=dtype)
    else:
        return np.array(np.hanning(size + 1)[:-1], dtype=dtype)


hann_window.support_native_out = False


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if periodic is False:
        return np.array(np.kaiser(M=window_length, beta=beta), dtype=dtype)
    else:
        return np.array(np.kaiser(M=window_length + 1, beta=beta)[:-1], dtype=dtype)


kaiser_window.support_native_out = False
