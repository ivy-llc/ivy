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
from ivy.functional.ivy.old import default_dtype
from ivy.functional.backends.numpy.device import _dev_callable

copy_array = lambda x: x.copy()
array_equal = _np.array_equal
to_numpy = lambda x: x
to_numpy.__name__ = 'to_numpy'
to_scalar = lambda x: x.item()
to_scalar.__name__ = 'to_scalar'
to_list = lambda x: x.tolist()
to_list.__name__ = 'to_list'

def is_array(x, exclusive=False):
    if isinstance(x, _np.ndarray):
        return True
    return False