# global
from typing import Optional, Tuple, Union
import math
import torch


# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_unsupported_device_and_dtypes,
)
from .. import backend_version

# noinspection PyProtectedMember


# Array API Standard #
# -------------------#


@with_unsupported_device_and_dtypes(
    {"2.0.1 and below": {"cpu": ("float16",)}},
    backend_version,
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
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


def unsorted_segment_min(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: Union[int, torch.Tensor],
) -> torch.Tensor:
    ivy.utils.assertions.check_unsorted_segment_min_valid_params(
        data, segment_ids, num_segments
    )
    if data.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        init_val = torch.finfo(data.dtype).max
    elif data.dtype in [torch.int32, torch.int64, torch.int8, torch.int16, torch.uint8]:
        init_val = torch.iinfo(data.dtype).max
    else:
        raise ValueError("Unsupported data type")

    res = torch.full(
        (num_segments,) + data.shape[1:], init_val, dtype=data.dtype, device=data.device
    )
    for i in range(num_segments):
        mask_index = segment_ids == i
        if torch.any(mask_index):
            res[i] = torch.min(data[mask_index], 0)[0]

    return res


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def blackman_window(
    size: int,
    /,
    *,
    periodic: bool = True,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.blackman_window(
        size,
        periodic=periodic,
        dtype=dtype,
    )


blackman_window.support_native_out = False


def unsorted_segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: Union[int, torch.Tensor],
) -> torch.Tensor:
    # Used the same check which is used for unsorted_segment_min as the
    # check should be same
    # Might require to change the assertion function name to
    # check_unsorted_segment_valid_params
    ivy.utils.assertions.check_unsorted_segment_min_valid_params(
        data, segment_ids, num_segments
    )

    res = torch.zeros(
        (num_segments,) + data.shape[1:], dtype=data.dtype, device=data.device
    )

    for i in range(num_segments):
        mask_index = segment_ids == i
        if torch.any(mask_index):
            res[i] = torch.sum(data[mask_index], dim=0)

    return res
