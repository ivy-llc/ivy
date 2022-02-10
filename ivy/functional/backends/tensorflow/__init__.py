import sys
import tensorflow as tf
from tensorflow.python.framework.dtypes import DType

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = Tensor
NativeVariable = Tensor
Device = str
Dtype = DType

# data types
int8 = tf.int8
int16 = tf.int16
int32 = tf.int32
int64 = tf.int64
uint8 = tf.uint8
uint16 = tf.uint16
uint32 = tf.uint32
uint64 = tf.uint64
bfloat16 = tf.bfloat16
float16 = tf.float16
float32 = tf.float32
float64 = tf.float64
# noinspection PyShadowingBuiltins
bool = tf.bool

all_dtypes = (int8, int16, int32, int64,
              uint8, uint16, uint32, uint64,
              bfloat16, float16, float32, float64)
valid_dtypes = all_dtypes
invalid_dtypes = ()

all_dtype_strs = ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'bfloat16', 'float16', 'float32', 'float64')
valid_dtype_strs = all_dtypes
invalid_dtype_strs = ()

iinfo = tf.experimental.numpy.iinfo

class Finfo:

    def __init__(self, tf_finfo):
        self._tf_finfo = tf_finfo

    @property
    def bits(self):
        return self._tf_finfo.bits

    @property
    def eps(self):
        return float(self._tf_finfo.eps)

    @property
    def max(self):
        return float(self._tf_finfo.max)

    @property
    def min(self):
        return float(self._tf_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._tf_finfo.tiny)


def finfo(datatype_in):
    return Finfo(tf.experimental.numpy.finfo(datatype_in))


backend = 'tensorflow'
