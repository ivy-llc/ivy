# global
import torch
from typing import Tuple
from collections import namedtuple


def unique_all(x : torch.Tensor, /)\
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    outputs, inverse_indices, counts = torch.unique(x, sorted=True, return_inverse=True,
                                                    return_counts=True, dim=None)
    
    Results = namedtuple(typename='unique_all', field_names=['values', 'indices', 'inverse_indices', 'counts'])
    flat_list = x.flatten().tolist()
    
    indices = [flat_list.index(val) for val in outputs]
    
    return Results(outputs.to(x.dtype), torch.tensor(indices).view(outputs.shape), inverse_indices, counts)