# global
import numpy as np
import tensorflow as tf
from typing import Union, Tuple
from tensorflow.python.types.core import Tensor
from tensorflow.python.framework.dtypes import DType

# local
import ivy


DTYPE_TO_STR = {tf.int8: 'int8',
                tf.int16: 'int16',
                tf.int32: 'int32',
                tf.int64: 'int64',
                tf.uint8: 'uint8',
                tf.uint16: 'uint16',
                tf.uint32: 'uint32',
                tf.uint64: 'uint64',
                tf.bfloat16: 'bfloat16',
                tf.float16: 'float16',
                tf.float32: 'float32',
                tf.float64: 'float64',
                tf.bool: 'bool'}

DTYPE_FROM_STR = {'int8': tf.int8,
                'int16': tf.int16,
                'int32': tf.int32,
                'int64': tf.int64,
                'uint8': tf.uint8,
                'uint16': tf.uint16,
                'uint32': tf.uint32,
                'uint64': tf.uint64,
                'bfloat16': tf.bfloat16,
                'float16': tf.float16,
                'float32': tf.float32,
                'float64': tf.float64,
                'bool': tf.bool}


# noinspection PyShadowingBuiltins
def iinfo(type: Union[DType, str, Tensor])\
        -> np.iinfo:
    return tf.experimental.numpy.iinfo(ivy.dtype_to_str(type))


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


# noinspection PyShadowingBuiltins
def finfo(type: Union[DType, str, Tensor])\
        -> Finfo:
    return Finfo(tf.experimental.numpy.finfo(ivy.dtype_from_str(type)))

  
def result_type(*arrays_and_dtypes: Union[Tensor, tf.DType]) -> tf.DType:
    if len(arrays_and_dtypes) <= 1:
        return tf.experimental.numpy.result_type(arrays_and_dtypes)

    result = tf.experimental.numpy.result_type(arrays_and_dtypes[0], arrays_and_dtypes[1])
    for i in range(2, len(arrays_and_dtypes)):
        result = tf.experimental.numpy.result_type(result, arrays_and_dtypes[i])
    return result


def broadcast_to (x: Tensor, shape: Tuple[int, ...])-> Tensor:
    return tf.broadcast_to(x, shape)

     
def astype(x: Tensor, dtype: tf.DType, copy: bool = True)\
        -> Tensor:
    if copy:
        if x.dtype == dtype:
            new_tensor = tf.experimental.numpy.copy(x)
            return new_tensor
    else:
        if x.dtype == dtype:
            return x
        else:
            new_tensor = tf.experimental.numpy.copy(x)
            new_tensor = tf.cast(new_tensor, dtype)
            return new_tensor
    return tf.cast(x, dtype)


def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace('tf.', '').replace('uint', '').replace('int', '').replace('bfloat', '').replace(
        'float', ''))


def dtype(x, as_str=False):
    dt = x.dtype
    if as_str:
        return dtype_to_str(dt)
    return dt


def dtype_to_str(dtype_in):
    if isinstance(dtype_in, str):
        return dtype_in
    return DTYPE_TO_STR[dtype_in]


def dtype_from_str(dtype_in):
    if not isinstance(dtype_in, str):
        return dtype_in
    return DTYPE_FROM_STR[dtype_in]
