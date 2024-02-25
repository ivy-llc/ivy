# global
import torch
from typing import Tuple, Optional
from collections import namedtuple

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
import ivy


@with_unsupported_dtypes(
    {
        "2.2 and below": ("complex", "float16"),
    },
    backend_version,
)
def unique_all(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    by_value: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Results = namedtuple(
        "Results",
        ["values", "indices", "inverse_indices", "counts"],
    )

    if axis is None:
        x = torch.flatten(x)
        axis = 0

    values, inverse_indices, counts = torch.unique(
        x,
        sorted=True,
        return_inverse=True,
        return_counts=True,
        dim=axis,
    )

    unique_nan = torch.isnan(values)
    idx_dtype = inverse_indices.dtype
    if torch.any(unique_nan):
        nan_index = torch.where(torch.isnan(x))
        non_nan_index = [
            x.tolist().index(val) for val in values if not torch.isnan(val)
        ]
        indices = values.clone().to(idx_dtype)
        indices[unique_nan] = nan_index[0]
        inverse_indices[torch.isnan(x)] = torch.where(unique_nan)[0][0]
        counts[unique_nan] = 1
        indices[~unique_nan] = torch.tensor(non_nan_index, dtype=idx_dtype)
    else:
        decimals = torch.arange(inverse_indices.numel()) / inverse_indices.numel()
        inv_sorted = (inverse_indices + decimals).argsort()
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        indices = inv_sorted[tot_counts].to(idx_dtype)

    if not by_value:
        sort_idx = torch.argsort(indices)
    else:
        values_ = torch.moveaxis(values, axis, 0)
        values_ = torch.reshape(values_, (values_.shape[0], -1))
        sort_idx = torch.tensor(
            [i[0] for i in sorted(enumerate(values_), key=lambda x: tuple(x[1]))]
        )
    ivy_torch = ivy.current_backend()
    values = values.index_select(dim=axis, index=sort_idx)
    counts = ivy_torch.gather(counts, sort_idx)
    indices = ivy_torch.gather(indices, sort_idx)
    inv_sort_idx = ivy_torch.invert_permutation(sort_idx)
    inverse_indices = torch.vmap(lambda y: torch.gather(inv_sort_idx, 0, y))(
        inverse_indices
    )

    return Results(
        values.to(x.dtype),
        indices,
        inverse_indices,
        counts,
    )


@with_unsupported_dtypes(
    {
        "2.2 and below": ("float16",),
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
        "2.2 and below": ("float16",),
    },
    backend_version,
)
def unique_inverse(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Results = namedtuple("Results", ["values", "inverse_indices"])

    if axis is None:
        x = torch.flatten(x)
        axis = 0

    values, inverse_indices = torch.unique(x, return_inverse=True, dim=axis)
    nan_idx = torch.isnan(x)
    if nan_idx.any():
        inverse_indices[nan_idx] = torch.where(torch.isnan(values))[0][0]
    inverse_indices = inverse_indices.reshape(x.shape)
    return Results(values, inverse_indices)


@with_unsupported_dtypes(
    {
        "2.2 and below": ("float16", "complex"),
    },
    backend_version,
)
def unique_values(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.unique(x)
