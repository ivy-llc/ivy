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

# Helpers #
# --------#

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

def _to_dev(x, dev):
    if dev is not None:
        if 'gpu' in dev:
            raise Exception('Native Numpy does not support GPU placement, consider using Jax instead')
        elif 'cpu' in dev:
            pass
        else:
            raise Exception('Invalid device specified, must be in the form [ "cpu:idx" | "gpu:idx" ],'
                            'but found {}'.format(dev))
    return x


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



def scatter_flat(indices, updates, size=None, tensor=None, reduction='sum', dev=None):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    if dev is None:
        dev = _dev_callable(updates)
    if reduction == 'sum':
        if not target_given:
            target = np.zeros([size], dtype=updates.dtype)
        np.add.at(target, indices, updates)
    elif reduction == 'replace':
        if not target_given:
            target = np.zeros([size], dtype=updates.dtype)
        target = np.asarray(target).copy()
        target.setflags(write=1)
        target[indices] = updates
    elif reduction == 'min':
        if not target_given:
            target = np.ones([size], dtype=updates.dtype) * 1e12
        np.minimum.at(target, indices, updates)
        if not target_given:
            target = np.where(target == 1e12, 0., target)
    elif reduction == 'max':
        if not target_given:
            target = np.ones([size], dtype=updates.dtype) * -1e12
        np.maximum.at(target, indices, updates)
        if not target_given:
            target = np.where(target == -1e12, 0., target)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    return _to_dev(target, dev)


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape=None, tensor=None, reduction='sum', dev=None):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.shape_to_tuple(target.shape) == ivy.shape_to_tuple(shape)
    if dev is None:
        dev = _dev_callable(updates)
    shape = list(shape) if ivy.exists(shape) else list(tensor.shape)
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    if reduction == 'sum':
        if not target_given:
            target = np.zeros(shape, dtype=updates.dtype)
        np.add.at(target, indices_tuple, updates)
    elif reduction == 'replace':
        if not target_given:
            target = np.zeros(shape, dtype=updates.dtype)
        target = np.asarray(target).copy()
        target.setflags(write=1)
        target[indices_tuple] = updates
    elif reduction == 'min':
        if not target_given:
            target = np.ones(shape, dtype=updates.dtype) * 1e12
        np.minimum.at(target, indices_tuple, updates)
        if not target_given:
            target = np.where(target == 1e12, 0., target)
    elif reduction == 'max':
        if not target_given:
            target = np.ones(shape, dtype=updates.dtype) * -1e12
        np.maximum.at(target, indices_tuple, updates)
        if not target_given:
            target = np.where(target == -1e12, 0., target)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    return _to_dev(target, dev)


def gather(params, indices, axis=-1, dev=None):
    if dev is None:
        dev = _dev_callable(params)
    return _to_dev(np.take_along_axis(params, indices, axis), dev)


def gather_nd(params, indices, dev=None):
    if dev is None:
        dev = _dev_callable(params)
    indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(_mul, params_shape[i + 1:], 1) for i in range(len(params_shape) - 1)] + [1]
    result_dim_sizes = np.array(result_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = np.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = np.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = np.tile(np.reshape(np.sum(indices * indices_scales, -1, keepdims=True), (-1, 1)), (1, implicit_indices_factor))
    implicit_indices = np.tile(np.expand_dims(np.arange(implicit_indices_factor), 0), (indices_for_flat_tiled.shape[0], 1))
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = np.reshape(indices_for_flat, (-1,)).astype(np.int32)
    flat_gather = np.take(flat_params, flat_indices_for_flat, 0)
    new_shape = list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    res = np.reshape(flat_gather, new_shape)
    return _to_dev(res, dev)

multiprocessing = lambda context=None: _multiprocessing if context is None else _multiprocessing.get_context(context)


def indices_where(x):
    where_x = np.where(x)
    if len(where_x) == 1:
        return np.expand_dims(where_x[0], -1)
    res = np.concatenate([np.expand_dims(item, -1) for item in where_x], -1)
    return res


# noinspection PyUnusedLocal
def one_hot(indices, depth, dev=None):
    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    res = np.eye(depth)[np.array(indices).reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])

shape = lambda x, as_tensor=False: np.asarray(np.shape(x)) if as_tensor else x.shape
shape.__name__ = 'shape'
get_num_dims = lambda x, as_tensor=False: np.asarray(len(np.shape(x))) if as_tensor else len(x.shape)