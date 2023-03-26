# global
from typing import Optional, Tuple
import math
import torch


# local
import ivy

# noinspection PyProtectedMember


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
    device: torch.device,
) -> Tuple[torch.Tensor]:
    n_cols = n_rows if n_cols is None else n_cols
    return tuple(
        torch.triu_indices(
            row=n_rows, col=n_cols, offset=k, dtype=torch.int64, device=device
        )
    )


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.kaiser_window(
        window_length,
        periodic,
        beta,
        dtype=dtype,
        layout=torch.strided,
        device=None,
        requires_grad=False,
    )


def hamming_window(
    window_length: int,
    /,
    *,
<<<<<<< HEAD
    periodic: Optional[bool] = True,
    alpha: Optional[float] = 0.54,
    beta: Optional[float] = 0.46,
=======
    periodic: bool = True,
    alpha: float = 0.54,
    beta: float = 0.46,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.hamming_window(
        window_length,
        periodic=periodic,
        alpha=alpha,
        beta=beta,
        dtype=dtype,
        layout=torch.strided,
        device=None,
        requires_grad=False,
    )


def vorbis_window(
    window_length: torch.tensor,
    *,
<<<<<<< HEAD
    dtype: Optional[torch.dtype] = torch.float32,
=======
    dtype: torch.dtype = torch.float32,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.tensor(
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


def hann_window(
    size: int,
    /,
    *,
<<<<<<< HEAD
    periodic: Optional[bool] = True,
=======
    periodic: bool = True,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.hann_window(
        size,
        periodic=periodic,
        dtype=dtype,
        layout=torch.strided,
        device=None,
        requires_grad=None,
    )


hann_window.support_native_out = False


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
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:

    n_cols = n_rows if n_cols is None else n_cols

    if n_rows <= 0 or n_cols <= 0:
        n_rows, n_cols = 0, 0

    return tuple(
        torch.tril_indices(
            row=n_rows, col=n_cols, offset=k, dtype=torch.int64, device=device
        )
    )
