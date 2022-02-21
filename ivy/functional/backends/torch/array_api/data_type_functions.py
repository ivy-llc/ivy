# global
import torch
from typing import Union

# local
import ivy


# noinspection PyShadowingBuiltins
def iinfo(type: Union[torch.dtype, str, torch.Tensor]) -> torch.iinfo:
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
def finfo(type: Union[torch.dtype, str, torch.Tensor]) -> Finfo:
    return Finfo(torch.finfo(ivy.dtype_from_str(type)))
