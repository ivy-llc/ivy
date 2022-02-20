# global
import sys
import torch as _torch

# local
from . import array_api
from .array_api import *
from . import array_builtins
from .array_builtins import *
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
bfloat16 = _torch.bfloat16
float16 = _torch.float16
float32 = _torch.float32
float64 = _torch.float64
# noinspection PyShadowingBuiltins
bool = _torch.bool

all_dtypes = (int8, int16, int32, int64,
              uint8,
              bfloat16, float16, float32, float64)
valid_dtypes = all_dtypes

all_dtype_strs = ('int8', 'int16', 'int32', 'int64',
                  'uint8',
                  'bfloat16', 'float16', 'float32', 'float64')
valid_dtype_strs = all_dtypes
invalid_dtype_strs = ('uint16', 'uint32', 'uint64')


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = dtype_to_str(type)
    if type_str in invalid_dtype_strs:
        return {'uint16': uint8,
                'uint32': uint8,
                'uint64': uint8}[type_str]
    return type


def iinfo(type):
    return _torch.iinfo(dtype_from_str(type))


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


def finfo(type):
    return Finfo(_torch.finfo(dtype_from_str(type)))


backend = 'torch'
