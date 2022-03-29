"""
Collection of TensorFlow general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
_round = round
import numpy as _np
import math as _math
import tensorflow as tf
from numbers import Number
from collections import Iterable

import multiprocessing as _multiprocessing
from tensorflow.python.types.core import Tensor

# local
from ivy.functional.ivy import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.tensorflow.device import _dev_callable, dev_from_str

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


# API #
# ----#







def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace('tf.', '').replace('uint', '').replace('int', '').replace('bfloat', '').replace(
        'float', ''))





minimum = tf.minimum
maximum = tf.maximum




def cast(x, dtype):
    return tf.cast(x, dtype_from_str(dtype))


astype = cast





















# noinspection PyShadowingNames
def zeros_like(x, dtype=None, dev=None):
    dtype = tf.__dict__[dtype] if dtype else dtype
    dev = default_device(dev)
    with tf.device(dev_from_str(dev)):
        return tf.zeros_like(x, dtype=dtype)


def full(shape, fill_value, dtype=None, device=None):
    with tf.device(dev_from_str(default_device(device))):
        return tf.fill(shape, tf.constant(fill_value, dtype=dtype_from_str(default_dtype(dtype, fill_value))))


def one_hot(indices, depth, dev=None):
    dev = default_device(dev)
    if dev is not None:
        with tf.device(dev_from_str(dev)):
            return tf.one_hot(indices, depth)
    return tf.one_hot(indices, depth)


cross = tf.linalg.cross




# noinspection PyShadowingNames
def identity(n, dtype='float32', batch_shape=None, dev=None):
    dtype = tf.__dict__[dtype]
    dev = default_device(dev)
    with tf.device(dev_from_str(dev)):
        return tf.eye(n, n, batch_shape=batch_shape, dtype=dtype)


meshgrid = lambda *xs, indexing='ij': tf.meshgrid(*xs, indexing=indexing)







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


compile = lambda fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None: tf.function(fn)
current_framework_str = lambda: 'tensorflow'
current_framework_str.__name__ = 'current_framework_str'

