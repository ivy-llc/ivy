# global
import numpy as np
import tensorflow as tf
from typing import Union
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
