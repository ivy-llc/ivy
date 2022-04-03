# global
import numpy as np
import tensorflow as tf
from typing import Union, Tuple
from tensorflow.python.types.core import Tensor
from tensorflow.python.framework.dtypes import DType

# local
import ivy


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
