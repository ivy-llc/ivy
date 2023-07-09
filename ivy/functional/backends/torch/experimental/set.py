# global
import torch
from typing import Tuple

# local
import ivy.functional.backends.torch as torch_backend
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes(
    {
        "2.0.1 and below": ("complex", "float16"),
    },
    backend_version,
)
def intersection(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    assume_unique: bool = False,
    return_indices: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = torch.reshape(x1, [-1])
    x2 = torch.reshape(x2, [-1])
    if not assume_unique:
        x1, ind1, _, _ = torch_backend.unique_all(x1)
        x2, ind2, _, _ = torch_backend.unique_all(x2)

    aux = torch.concatenate((x1, x2))
    if return_indices:
        aux_sort_indices = torch.argsort(aux, stable=True)
        aux = aux[aux_sort_indices]
    else:
        aux, _ = aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]
    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - x1.shape[0]
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices.to(torch.int64), ar2_indices.to(torch.int64)
    else:
        return int1d
