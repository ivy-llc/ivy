import sys
import mxnet as mx
import numpy as np

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
int8 = np.dtype('int8')
int16 = np.dtype('int16')
int32 = np.dtype('int32')
int64 = np.dtype('int64')
uint8 = np.dtype('uint8')
uint16 = np.dtype('uint16')
uint32 = np.dtype('uint32')
uint64 = np.dtype('uint64')
bfloat16 = 'bfloat16'
float16 = np.dtype('float16')
float32 = np.dtype('float32')
float64 = np.dtype('float64')
# noinspection PyShadowingBuiltins
bool = np.dtype('bool')

iinfo = np.iinfo

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


def finfo(datatype_in):
    return Finfo(np.finfo(datatype_in))


backend = 'mxnet'
