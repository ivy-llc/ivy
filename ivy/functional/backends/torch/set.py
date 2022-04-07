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
  
  
def unique_inverse(x: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    values, inverse_indices = torch.unique(x, return_inverse=True)
    nan_idx = torch.isnan(x)
    if nan_idx.any():
        inverse_indices[nan_idx] = torch.where(torch.isnan(values))[0][0]
    inverse_indices = inverse_indices.reshape(x.shape)
    return out(values, inverse_indices)


def unique_values(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.unique(x)


def unique_counts(x: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    v, c = torch.unique(torch.reshape(x, [-1]), return_counts=True)
    nan_idx = torch.where(torch.isnan(v))
    c[nan_idx] = 1
    uc = namedtuple('uc', ['values', 'counts'])
    return uc(v, c)
