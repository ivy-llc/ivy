import sys
import ivy
import numpy as np

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = np.ndarray
NativeVariable = np.ndarray
Device = str
Dtype = np.dtype

# data types
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64
bfloat16 = np.bfloat16
float16 = np.float16
float32 = np.float32
float64 = np.float64
# noinspection PyShadowingBuiltins
bool = np.bool_

backend = 'numpy'
