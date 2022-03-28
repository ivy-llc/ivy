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







@_handle_flat_arrays_in_out
def concatenate(xs, axis=-1):
    return _mx.nd.concat(*xs, dim=axis)





@_handle_flat_arrays_in_out
def where(condition, x1, x2):
    x_shape = list(x1.shape)
    condition_shape = list(condition.shape)
    if x_shape == condition_shape:
        res = _mx.nd.where(condition, x1, x2)
        return res
    tile_reps = [int(x / c) for x, c in zip(x_shape, condition_shape)]
    tiled_condition = _mx.nd.tile(condition, tile_reps)
    return _mx.nd.where(tiled_condition, x1, x2)



reshape = lambda x, new_shape: x.reshape(new_shape)


def broadcast_to(x, new_shape):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    num_shape_dims = len(new_shape)
    diff = num_shape_dims - num_x_dims
    if diff == 0:
        return _mx.nd.broadcast_to(x, new_shape)
    x = _mx.nd.reshape(x, [1]*diff + x_shape)
    return _mx.nd.broadcast_to(x, new_shape)


def squeeze(x, axis=None):
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise Exception('tried to squeeze a zero-dimensional input by axis {}'.format(axis))
    res = _mx.nd.squeeze(x, axis)
    if axis is None:
        return _1_dim_array_to_flat_array(res)
    return res


# noinspection PyShadowingNames



def zeros_like(x, dtype=None, dev=None):
    if x.shape == ():
        return _mx.nd.array(0., ctx=_mxnet_init_context(default_device(dev)))
    mx_zeros = _mx.nd.zeros_like(x, ctx=_mxnet_init_context(default_device(dev)))
    return mx_zeros if not dtype else mx_zeros.astype(dtype)


def full(shape, fill_value, dtype=None, device=None):
    shape = ivy.shape_to_tuple(shape)
    cont = _mxnet_init_context(default_device(device))
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(
            _mx.nd.full((1,), fill_value, cont, dtype_from_str(default_dtype(dtype, fill_value))))
    return _mx.nd.full(shape, fill_value, cont, dtype_from_str(default_dtype(dtype, fill_value)))



def cross(x1, x2):
    a1 = x1[..., 0:1]
    a2 = x1[..., 1:2]
    a3 = x1[..., 2:3]
    b1 = x2[..., 0:1]
    b2 = x2[..., 1:2]
    b3 = x2[..., 2:3]
    res1 = a2*b3 - a3*b2
    res2 = a3*b1 - a1*b3
    res3 = a1*b2 - a2*b1
    res = _mx.nd.concat(res1, res2, res3, dim=-1)
    return res


def matmul(x1, x2):
    expanded = False
    x1_shape = list(x1.shape)
    x2_shape = list(x2.shape)
    if len(x1_shape) != 3:
        num_x1_dims = len(x1_shape)
        x1 = _mx.nd.reshape(x1, [1]*max(2-num_x1_dims, 0) + [-1] + x1_shape[-min(num_x1_dims, 2):])
        expanded = True
    if len(x2_shape) != 3:
        num_x2_dims = len(x2_shape)
        x2 = _mx.nd.reshape(x2, [1]*max(2-num_x2_dims, 0) + [-1] + x2_shape[-min(num_x2_dims, 2):])
        expanded = True
    x1_batch_size = x1.shape[0]
    x2_batch_size = x2.shape[0]
    if x1_batch_size > x2_batch_size:
        x2 = _mx.nd.tile(x2, (int(x1_batch_size/x2_batch_size), 1, 1))
    elif x2_batch_size > x1_batch_size:
        x1 = _mx.nd.tile(x1, (int(x2_batch_size / x1_batch_size), 1, 1))
    res = _mx.nd.batch_dot(x1, x2)
    if expanded:
        return _mx.nd.reshape(res, list(x1_shape[:-1]) + [res.shape[-1]])
    return res




def identity(n, dtype='float32', batch_shape=None, dev=None):
    mat = _mx.nd.eye(n, dtype=dtype).copyto(_mxnet_init_context(default_device(dev)))
    if batch_shape is None:
        return mat
    else:
        reshape_dims = [1]*len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        res = _mx.nd.tile(_mx.nd.reshape(mat, reshape_dims), tile_dims)
        return res


def meshgrid(*xs, indexing='ij'):
    # ToDo: implement this without reliance on NumPy backend
    xs_np = [x.as_np_ndarray() for x in xs]
    return tuple([item.as_nd_ndarray() for item in _mx.np.meshgrid(*xs_np, indexing=indexing)])




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








