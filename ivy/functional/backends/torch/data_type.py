# global
import torch
from typing import Union, Tuple

# local
import ivy


# noinspection PyShadowingBuiltins
def iinfo(type: Union[torch.dtype, str, torch.Tensor])\
        -> torch.iinfo:
    return torch.iinfo(ivy.dtype_from_str(type))


class Finfo:

    def __init__(self, torch_finfo):
        self._torch_finfo = torch_finfo

    @property
    def bits(self):
        return self._torch_finfo.bits

    @property
    def eps(self):
        return self._torch_finfo.eps

    @property
    def max(self):
        return self._torch_finfo.max

    @property
    def min(self):
        return self._torch_finfo.min

    @property
    def smallest_normal(self):
        return self._torch_finfo.tiny


# noinspection PyShadowingBuiltins
def finfo(type: Union[torch.dtype, str, torch.Tensor])\
        -> Finfo:
    return Finfo(torch.finfo(ivy.dtype_from_str(type)))


def result_type(*arrays_and_dtypes: Union[torch.tensor, torch.dtype]) -> torch.dtype:
    arrays_and_dtypes = list(arrays_and_dtypes)
    for i in range(len(arrays_and_dtypes)):
        if type(arrays_and_dtypes[i]) == torch.dtype:
            arrays_and_dtypes[i] = torch.tensor([], dtype=arrays_and_dtypes[i])
    if len(arrays_and_dtypes) == 1:
        return arrays_and_dtypes[0].dtype
    result = torch.result_type(arrays_and_dtypes[0], arrays_and_dtypes[1])
    for i in range(2, len(arrays_and_dtypes)):
        result = torch.result_type(torch.tensor([], dtype=result), arrays_and_dtypes[i])
    return result

  
def broadcast_to(x: torch.Tensor, shape: Tuple[int,...]) -> torch.Tensor:
    return torch.broadcast_to(x,shape)


def astype(x: torch.Tensor, dtype: torch.dtype, copy: bool = True)\
     -> torch.Tensor:
    if copy:
        if x.dtype == dtype:
            new_tensor = x.clone().detach()
            return new_tensor
    else:
        if x.dtype == dtype:
            return x
        else:
            new_tensor = x.clone().detach()
            return new_tensor.to(dtype)
    return x.to(dtype)
