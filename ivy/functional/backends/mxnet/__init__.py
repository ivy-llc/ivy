# global
import sys
import mxnet as mx
import numpy as np

# local
from . import array_api
from .array_api import *
from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = mx.ndarray.ndarray.NDArray
NativeVariable = mx.ndarray.ndarray.NDArray
Device = mx.context.Context
Dtype = type

# data types
int8 = np.int8
int32 = np.int32
int64 = np.int64
uint8 = np.uint8
float16 = np.float16
float32 = np.float32
float64 = np.float64
# noinspection PyShadowingBuiltins
bool = np.bool

all_dtypes = (int8, int32, int64,
              uint8,
              float16, float32, float64)
valid_dtypes = all_dtypes

all_dtype_strs = ('int8', 'int32', 'int64',
                  'uint8',
                  'float16', 'float32', 'float64')
valid_dtype_strs = all_dtypes
invalid_dtype_strs = ('int16', 'uint16', 'uint32', 'uint64', 'bfloat16')


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = dtype_to_str(type)
    if type_str in invalid_dtype_strs:
        return {'int16': int32,
                'uint16': uint8,
                'uint32': uint8,
                'uint64': uint8,
                'bfloat16': float16}[type_str]
    return type


def iinfo(type):
    return np.iinfo(dtype_from_str(type))


class Finfo:

    def __init__(self, mx_finfo):
        self._mx_finfo = mx_finfo

    @property
    def bits(self):
        return self._mx_finfo.bits

    @property
    def eps(self):
        return float(self._mx_finfo.eps)

    @property
    def max(self):
        return float(self._mx_finfo.max)

    @property
    def min(self):
        return float(self._mx_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._mx_finfo.tiny)


def finfo(type):
    return Finfo(np.finfo(dtype_from_str(type)))


backend = 'mxnet'
