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







def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace('uint', '').replace('int', '').replace('bfloat', '').replace('float', ''))




shape = lambda x, as_tensor=False: np.asarray(np.shape(x)) if as_tensor else x.shape
shape.__name__ = 'shape'
get_num_dims = lambda x, as_tensor=False: np.asarray(len(np.shape(x))) if as_tensor else len(x.shape)
minimum = np.minimum
maximum = np.maximum
clip = lambda x, x_min, x_max: np.asarray(np.clip(x, x_min, x_max))
abs = lambda x: np.asarray(np.absolute(x))


def argmin(x, axis=0):
    ret = np.asarray(np.argmin(x, axis))
    if ret.shape == ():
        return ret.reshape(-1)
    return ret


def cast(x, dtype):
    return x.astype(dtype_from_str(dtype))


astype = cast


# noinspection PyShadowingNames
def arange(stop, start=0, step=1, dtype=None, dev=None):
    if dtype:
        dtype = dtype_from_str(dtype)
    res = _to_dev(np.arange(start, stop, step=step, dtype=dtype), dev)
    if not dtype:
        if res.dtype == np.float64:
            return res.astype(np.float32)
        elif res.dtype == np.int64:
            return res.astype(np.int32)
    return res




def concatenate(xs, axis=-1):
    if xs[0].shape == ():
        return np.concatenate([np.expand_dims(x, 0) for x in xs], axis)
    return np.concatenate(xs, axis)


stack = np.stack




def transpose(x, axes=None):
    if axes is None:
        num_dims = len(x.shape)
        axes = list(range(num_dims))
        axes.reverse()
    return np.transpose(x, axes)


where = lambda condition, x1, x2: np.where(condition, x1, x2)


def indices_where(x):
    where_x = np.where(x)
    if len(where_x) == 1:
        return np.expand_dims(where_x[0], -1)
    res = np.concatenate([np.expand_dims(item, -1) for item in where_x], -1)
    return res


reshape = np.reshape
broadcast_to = np.broadcast_to


def squeeze(x, axis=None):
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise Exception('tried to squeeze a zero-dimensional input by axis {}'.format(axis))
    return np.squeeze(x, axis)




# noinspection PyShadowingNames
def zeros_like(x, dtype=None, dev=None):
    if dtype:
        dtype = 'bool_' if dtype == 'bool' else dtype
        dtype = np.__dict__[dtype]
    else:
        dtype = x.dtype
    return _to_dev(np.zeros_like(x, dtype=dtype), dev)


def full(shape, fill_value, dtype=None, device=None):
    return _to_dev(np.full(shape, fill_value, dtype_from_str(default_dtype(dtype, fill_value))), device)


# noinspection PyUnusedLocal
def one_hot(indices, depth, dev=None):
    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    res = np.eye(depth)[np.array(indices).reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


cross = np.cross



# noinspection PyShadowingNames
def identity(n, dtype='float32', batch_shape=None, dev=None):
    dtype = 'bool_' if dtype == 'bool' else dtype
    dtype = np.__dict__[dtype]
    mat = np.identity(n, dtype=dtype)
    if batch_shape is None:
        return_mat = mat
    else:
        reshape_dims = [1] * len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        return_mat = np.tile(np.reshape(mat, reshape_dims), tile_dims)
    return _to_dev(return_mat, dev)


meshgrid = lambda *xs, indexing='ij': np.meshgrid(*xs, indexing=indexing)


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


def linear_resample(x, num_samples, axis=-1):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    axis = axis % num_x_dims
    x_pre_shape = x_shape[0:axis]
    x_pre_size = _reduce(_mul, x_pre_shape) if x_pre_shape else 1
    num_pre_dims = len(x_pre_shape)
    num_vals = x.shape[axis]
    x_post_shape = x_shape[axis+1:]
    x_post_size = _reduce(_mul, x_post_shape) if x_post_shape else 1
    num_post_dims = len(x_post_shape)
    xp = np.reshape(np.arange(num_vals*x_pre_size*x_post_size), x_shape)
    x_coords = np.arange(num_samples) * ((num_vals-1)/(num_samples-1)) * x_post_size
    x_coords = np.reshape(x_coords, [1]*num_pre_dims + [num_samples] + [1]*num_post_dims)
    x_coords = np.broadcast_to(x_coords, x_pre_shape + [num_samples] + x_post_shape)
    slc = [slice(None)] * num_x_dims
    slc[axis] = slice(0, 1, 1)
    x_coords = x_coords + xp[tuple(slc)]
    x = np.reshape(x, (-1,))
    xp = np.reshape(xp, (-1,))
    x_coords = np.reshape(x_coords, (-1,))
    ret = np.interp(x_coords, xp, x)
    return np.reshape(ret, x_pre_shape + [num_samples] + x_post_shape)


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


# noinspection PyUnusedLocal
def compile(func, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None):
    logging.warning('Numpy does not support compiling functions.\n'
                    'Now returning the unmodified function.')
    return func


current_framework_str = lambda: 'numpy'
current_framework_str.__name__ = 'current_framework_str'
multiprocessing = lambda context=None: _multiprocessing if context is None else _multiprocessing.get_context(context)



