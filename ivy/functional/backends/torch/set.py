# global
import torch
from typing import Tuple
from collections import namedtuple

# local
from ivy.functional.backends.torch import Tensor


def unique_all(x : Tensor)\
                -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    x_ = x.clone()

    if x.requires_grad_:
        x_ = x_.detach()

    outputs, inverse_indices, counts = torch.unique(x_, sorted=True, return_inverse=True,
                                                    return_counts=True, dim=None)
    indices = list()
    Results = namedtuple(typename='unique_all', field_names='values indices inverse_indices counts')

    for value in outputs:
        r, c = torch.where(x_ == value)
        indices.append([r[0], c[0]])

    return Results(outputs.to(x_.dtype), torch.tensor(indices), inverse_indices, counts)