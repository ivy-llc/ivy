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
from ivy.functional.backends.tensorflow.device import _dev_callable, dev_from_str

def is_array(x, exclusive=False):
    if isinstance(x, Tensor):
        if exclusive and isinstance(x, _tf.Variable):
            return False
        return True
    return False


copy_array = _tf.identity
