"""
Collection of Numpy general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import logging
import numpy as np
import math as _math
from operator import mul as _mul
from functools import reduce as _reduce
import multiprocessing as _multiprocessing

# local
import ivy
from ivy.functional.ivy import default_dtype
from ivy.functional.backends.numpy.device import _dev_callable, to_dev


copy_array = lambda x: x.copy()
array_equal = np.array_equal
floormod = lambda x, y: np.asarray(x % y)

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
    if isinstance(x, np.ndarray):
        return True
    return False


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    x_split = np.split(x, x.shape[axis], axis)
    if keepdims:
        return x_split
    return [np.squeeze(item, axis) for item in x_split]


def inplace_decrement(x, val):
    x -= val
    return x


def inplace_increment(x, val):
    x += val
    return x

cumsum = np.cumsum

def cumprod(x, axis=0, exclusive=False):
    if exclusive:
        x = np.swapaxes(x, axis, -1)
        x = np.concatenate((np.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = np.cumprod(x, -1)
        return np.swapaxes(res, axis, -1)
    return np.cumprod(x, axis)
