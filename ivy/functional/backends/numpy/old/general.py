"""
Collection of Numpy general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import logging
import numpy as np
import math as _math

import multiprocessing as _multiprocessing

# local
import ivy
from ivy.functional.ivy import default_dtype
from ivy.functional.backends.numpy.device import _dev_callable
#temporary
from ivy.functional.backends.numpy.general import _to_dev

DTYPE_TO_STR = {np.dtype('int8'): 'int8',
                np.dtype('int16'): 'int16',
                np.dtype('int32'): 'int32',
                np.dtype('int64'): 'int64',
                np.dtype('uint8'): 'uint8',
                np.dtype('uint16'): 'uint16',
                np.dtype('uint32'): 'uint32',
                np.dtype('uint64'): 'uint64',
                'bfloat16': 'bfloat16',
                np.dtype('float16'): 'float16',
                np.dtype('float32'): 'float32',
                np.dtype('float64'): 'float64',
                np.dtype('bool'): 'bool',

                np.int8: 'int8',
                np.int16: 'int16',
                np.int32: 'int32',
                np.int64: 'int64',
                np.uint8: 'uint8',
                np.uint16: 'uint16',
                np.uint32: 'uint32',
                np.uint64: 'uint64',
                np.float16: 'float16',
                np.float32: 'float32',
                np.float64: 'float64',
                np.bool_: 'bool'}

DTYPE_FROM_STR = {'int8': np.dtype('int8'),
                'int16': np.dtype('int16'),
                'int32': np.dtype('int32'),
                'int64': np.dtype('int64'),
                'uint8': np.dtype('uint8'),
                'uint16': np.dtype('uint16'),
                'uint32': np.dtype('uint32'),
                'uint64': np.dtype('uint64'),
                'bfloat16': 'bfloat16',
                'float16': np.dtype('float16'),
                'float32': np.dtype('float32'),
                'float64': np.dtype('float64'),
                'bool': np.dtype('bool')}




# API #
# ----#









