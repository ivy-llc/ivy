# global
from typing import Optional, Tuple
import math
import torch


# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
)
from .. import backend_version

# noinspection PyProtectedMember


# Array API Standard #
# -------------------#


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
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
    periodic: bool = True,
    alpha: float = 0.54,
    beta: float = 0.46,
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
    dtype: torch.dtype = torch.float32,
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


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def hann_window(
    size: int,
    /,
    *,
    periodic: bool = True,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.hann_window(
        size,
        periodic=periodic,
        dtype=dtype,
    )


hann_window.support_native_out = False


def tril_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
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
