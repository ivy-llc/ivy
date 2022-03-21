"""
Collection of Numpy general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import logging
import numpy as _np
import math as _math
from operator import mul as _mul
from functools import reduce as _reduce
import multiprocessing as _multiprocessing

# local
import ivy
from ivy.functional.ivy import default_dtype
from ivy.functional.backends.numpy.device import _dev_callable, to_dev


copy_array = lambda x: x.copy()
array_equal = _np.array_equal
floormod = lambda x, y: _np.asarray(x % y)

to_numpy = lambda x: x
to_numpy.__name__ = 'to_numpy'
to_scalar = lambda x: x.item()
to_scalar.__name__ = 'to_scalar'
to_list = lambda x: x.tolist()
to_list.__name__ = 'to_list'
container_types = lambda: []
inplace_arrays_supported = lambda: True
inplace_variables_supported = lambda: True



def inplace_update(x, val):
    x.data = val
    return x


def is_array(x, exclusive=False):
    if isinstance(x, _np.ndarray):
        return True
    return False


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    x_split = _np.split(x, x.shape[axis], axis)
    if keepdims:
        return x_split
    return [_np.squeeze(item, axis) for item in x_split]


def inplace_decrement(x, val):
    x -= val
    return x


def inplace_increment(x, val):
    x += val
    return x


