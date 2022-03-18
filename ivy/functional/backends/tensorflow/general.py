"""
Collection of TensorFlow general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
_round = round
import numpy as _np
import math as _math
import tensorflow as _tf
from numbers import Number
import tensorflow_probability as _tfp
import multiprocessing as _multiprocessing
from tensorflow.python.types.core import Tensor

# local
from ivy.functional.ivy.old import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.tensorflow import linspace
from ivy.functional.backends.tensorflow.device import _dev_callable, dev_from_str

def is_array(x, exclusive=False):
    if isinstance(x, Tensor):
        if exclusive and isinstance(x, _tf.Variable):
            return False
        return True
    return False


copy_array = _tf.identity
array_equal = _tf.experimental.numpy.array_equal
floormod = lambda x, y: x % y
to_numpy = lambda x: _np.asarray(_tf.convert_to_tensor(x))
to_numpy.__name__ = 'to_numpy'
to_scalar = lambda x: to_numpy(x).item()
to_scalar.__name__ = 'to_scalar'
to_list = lambda x: x.numpy().tolist()
to_list.__name__ = 'to_list'


def logspace(start, stop, num, base=10., axis=None, dev=None):
    power_seq = linspace(start, stop, num, axis, default_device(dev))
    return base ** power_seq


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    ret = _tf.unstack(x, axis=axis)
    if keepdims:
        return [_tf.expand_dims(r, axis) for r in ret]
    return ret
