import sys
import torch as _torch

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = _torch.Tensor
NativeVariable = _torch.Tensor
Device = _torch.device
Dtype = _torch.dtype

# data types
int8 = _torch.int8
int16 = _torch.int16
int32 = _torch.int32
int64 = _torch.int64
uint8 = _torch.uint8
uint16 = 'uint16'
uint32 = 'uint32'
uint64 = 'uint64'
bfloat16 = _torch.bfloat16
float16 = _torch.float16
float32 = _torch.float32
float64 = _torch.float64
# noinspection PyShadowingBuiltins
bool = _torch.bool

iinfo = _torch.iinfo

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


def finfo(datatype_in):
    return Finfo(_torch.finfo(datatype_in))


backend = 'torch'
