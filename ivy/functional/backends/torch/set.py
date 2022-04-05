# global
import torch
from typing import Tuple
from collections import namedtuple

# local
from ivy.functional.backends.torch import Tensor


def unique_all(x: torch.Tensor, /) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    outputs, inverse_indices, counts = torch.unique(x, sorted=False, return_inverse=True,
                                                    return_counts=True, dim=None)
    indices = list()
    Results = namedtuple(typename='unique_all', field_names=['values', 'indices', 'inverse_indices', 'counts'])
    temp = list()
    for val in iter(outputs):
        loc = torch.where(x == val)
        for ix in iter(loc):
            temp.append(ix[0])
        indices.append(temp)
        temp.clear()

    return Results(outputs.to(x.dtype), torch.tensor(indices), inverse_indices, counts)