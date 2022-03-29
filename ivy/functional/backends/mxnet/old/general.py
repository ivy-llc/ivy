"""
Collection of MXNet general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
_round = round
import logging
import mxnet as _mx
import numpy as _np
import math as _math
from numbers import Number
from operator import mul as _mul
from functools import reduce as _reduce
import multiprocessing as _multiprocessing

# local
from ivy.functional.ivy import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.mxnet.device import _callable_dev
from ivy.functional.backends.mxnet.general import unstack
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out, _mxnet_init_context,\
    _scalar_or_flat_array_to_scalar, _handle_flat_arrays_in, _flat_array_to_1_dim_array, _1_dim_array_to_flat_array

#temporary imports
from ivy.functional.backends.mxnet.general import linspace


DTYPE_TO_STR = {_np.dtype('int8'): 'int8',
                _np.dtype('int16'): 'int16',
                _np.dtype('int32'): 'int32',
                _np.dtype('int64'): 'int64',
                _np.dtype('uint8'): 'uint8',
                _np.dtype('uint16'): 'uint16',
                _np.dtype('uint32'): 'uint32',
                _np.dtype('uint64'): 'uint64',
                'bfloat16': 'bfloat16',
                _np.dtype('float16'): 'float16',
                _np.dtype('float32'): 'float32',
                _np.dtype('float64'): 'float64',
                _np.dtype('bool'): 'bool',

                _np.int8: 'int8',
                _np.int16: 'int16',
                _np.int32: 'int32',
                _np.int64: 'int64',
                _np.uint8: 'uint8',
                _np.uint16: 'uint16',
                _np.uint32: 'uint32',
                _np.uint64: 'uint64',
                _np.float16: 'float16',
                _np.float32: 'float32',
                _np.float64: 'float64',
                _np.bool_: 'bool'}

DTYPE_FROM_STR = {'int8': _np.int8,
                'int16': _np.int16,
                'int32': _np.int32,
                'int64': _np.int64,
                'uint8': _np.uint8,
                'uint16': _np.uint16,
                'uint32': _np.uint32,
                'uint64': _np.uint64,
                'bfloat16': 'bfloat16',
                'float16': _np.float16,
                'float32': _np.float32,
                'float64': _np.float64,
                'bool': _np.bool_}


# API #
# ----#









def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace("<class 'numpy.", '').replace("'>", '').replace('uint', '').replace(
        'int', '').replace('bfloat', '').replace('float', ''))




minimum = lambda x, y: _mx.nd.array(_mx.nd.minimum(_scalar_or_flat_array_to_scalar(x), _scalar_or_flat_array_to_scalar(y)))
maximum = lambda x, y: _mx.nd.array(_mx.nd.maximum(_scalar_or_flat_array_to_scalar(x), _scalar_or_flat_array_to_scalar(y)))




# noinspection PyShadowingBuiltins
@_handle_flat_arrays_in_out
def abs(x):
    return _mx.nd.abs(x)


@_handle_flat_arrays_in_out
def cast(x, dtype):
    return x.astype(dtype)


astype = cast















# noinspection PyShadowingNames










def identity(n, dtype='float32', batch_shape=None, dev=None):
    mat = _mx.nd.eye(n, dtype=dtype).copyto(_mxnet_init_context(default_device(dev)))
    if batch_shape is None:
        return mat
    else:
        reshape_dims = [1]*len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        res = _mx.nd.tile(_mx.nd.reshape(mat, reshape_dims), tile_dims)
        return res





def dtype(x, as_str=False):
    dt = x.dtype
    if as_str:
        return dtype_to_str(dt)
    return x.dtype


def dtype_to_str(dtype_in):
    if isinstance(dtype_in, str):
        return dtype_in
    return DTYPE_TO_STR[dtype_in]


def dtype_from_str(dtype_in):
    if not isinstance(dtype_in, str):
        return dtype_in
    return DTYPE_FROM_STR[dtype_in]


# noinspection PyUnusedLocal
def compile(func, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None):
    logging.warning('MXnet does not support compiling arbitrary functions, '
                    'consider writing a function using MXNet Symbolic backend instead for compiling.\n'
                    'Now returning the unmodified function.')
    return func


current_framework_str = lambda: 'mxnet'
current_framework_str.__name__ = 'current_framework_str'








