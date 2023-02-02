# global
import torch
from typing import Tuple, Optional
from collections import namedtuple

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes(
    {
        "1.11.0 and below": ("float16", "complex"),
    },
    backend_version,
)
def unique_all(
    x: torch.Tensor,
    /,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Results = namedtuple(
        "Results",
        ["values", "indices", "inverse_indices", "counts"],
    )

    outputs, inverse_indices, counts = torch.unique(
        x, sorted=True, return_inverse=True, return_counts=True, dim=None
    )

    flat_tensor = x.flatten()
    unique_nan = torch.isnan(outputs)
    idx_dtype = inverse_indices.dtype

    if torch.any(unique_nan):
        nan_index = torch.where(torch.isnan(flat_tensor))
        non_nan_index = [
            flat_tensor.tolist().index(val) for val in outputs if not torch.isnan(val)
        ]

        indices = outputs.clone().to(idx_dtype)

        indices[unique_nan] = nan_index[0]
        inverse_indices[torch.isnan(x)] = torch.where(unique_nan)[0][0]
        counts[unique_nan] = 1
        indices[~unique_nan] = torch.tensor(non_nan_index, dtype=idx_dtype)

    else:
        indices = torch.tensor(
            [torch.where(flat_tensor == val)[0][0] for val in outputs], dtype=idx_dtype
        )

    return Results(
        outputs.to(x.dtype),
        indices.view(outputs.shape),
        inverse_indices.reshape(x.shape),
        counts,
    )


@with_unsupported_dtypes(
    {
        "1.11.0 and below": ("float16",),
    },
    backend_version,
)
def unique_counts(x: torch.Tensor, /) -> Tuple[torch.Tensor, torch.Tensor]:
    v, c = torch.unique(torch.reshape(x, [-1]), return_counts=True)
    nan_idx = torch.where(torch.isnan(v))
    c[nan_idx] = 1
    Results = namedtuple("Results", ["values", "counts"])
    return Results(v, c)


@with_unsupported_dtypes(
    {
        "1.11.0 and below": ("float16",),
    },
    backend_version,
)
def unique_inverse(x: torch.Tensor, /) -> Tuple[torch.Tensor, torch.Tensor]:
    Results = namedtuple("Results", ["values", "inverse_indices"])
    values, inverse_indices = torch.unique(x, return_inverse=True)
    nan_idx = torch.isnan(x)
    if nan_idx.any():
        inverse_indices[nan_idx] = torch.where(torch.isnan(values))[0][0]
    inverse_indices = inverse_indices.reshape(x.shape)
    return Results(values, inverse_indices)


@with_unsupported_dtypes(
    {
        "1.11.0 and below": ("float16", "complex"),
    },
    backend_version,
)
def unique_values(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.unique(x)
