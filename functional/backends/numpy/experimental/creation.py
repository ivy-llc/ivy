# global
from typing import Optional, Tuple

import numpy as np
import math

# local
from ivy.functional.backends.numpy.device import _to_device
import ivy

# Array API Standard #
# -------------------#


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
<<<<<<< HEAD
    k: Optional[int] = 0,
=======
    k: int = 0,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
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
<<<<<<< HEAD
    dtype: Optional[np.dtype] = np.float32,
=======
    dtype: np.dtype = np.float32,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.array(
        [
            round(
                math.sin(
                    (ivy.pi / 2) * (math.sin(ivy.pi * (i) / (window_length * 2)) ** 2)
                ),
                8,
            )
            for i in range(1, window_length * 2)[0::2]
        ],
        dtype=dtype,
    )


vorbis_window.support_native_out = False


def tril_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
<<<<<<< HEAD
    k: Optional[int] = 0,
=======
    k: int = 0,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
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
<<<<<<< HEAD
    periodic: Optional[bool] = True,
=======
    periodic: bool = True,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    size = size + 1 if periodic is True else size
    return np.array(np.hanning(size), dtype=dtype)


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
