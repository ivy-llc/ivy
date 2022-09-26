# global
from collections import namedtuple
from typing import NamedTuple, Optional

import torch

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import version


@with_unsupported_dtypes(
    {
        "1.11.0 and below": ("float16",),
    },
    version,
)
def unique_all(
    x: torch.Tensor,
    /,
) -> NamedTuple:
    Results = namedtuple(
        typename="unique_all",
        field_names=["values", "indices", "inverse_indices", "counts"],
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
    version,
)
def unique_counts(x: torch.Tensor, /) -> NamedTuple:
    v, c = torch.unique(torch.reshape(x, [-1]), return_counts=True)
    nan_idx = torch.where(torch.isnan(v))
    c[nan_idx] = 1
    uc = namedtuple("uc", ["values", "counts"])
    return uc(v, c)


@with_unsupported_dtypes(
    {
        "1.11.0 and below": ("float16",),
    },
    version,
)
def unique_inverse(x: torch.Tensor, /) -> NamedTuple:
    out = namedtuple("unique_inverse", ["values", "inverse_indices"])
    values, inverse_indices = torch.unique(x, return_inverse=True)
    nan_idx = torch.isnan(x)
    if nan_idx.any():
        inverse_indices[nan_idx] = torch.where(torch.isnan(values))[0][0]
    inverse_indices = inverse_indices.reshape(x.shape)
    return out(values, inverse_indices)


@with_unsupported_dtypes(
    {
        "1.11.0 and below": ("float16",),
    },
    version,
)
def unique_values(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.unique(x)
