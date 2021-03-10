"""
Collection of Numpy general functions, wrapped to fit Ivy syntax and signature.
"""

# global
from functools import reduce as _reduce
from operator import mul as _mul
import numpy as _np
import logging

DTYPE_DICT = {_np.dtype('int32'): 'int32',
              _np.dtype('int64'): 'int64',
              _np.dtype('float32'): 'float32',
              _np.dtype('float64'): 'float64'}


def _to_dev(x, dev):
    if dev is not None:
        if 'gpu' in dev:
            raise Exception('Native Numpy does not support GPU placement, consider using Jax instead')
        elif 'cpu' in dev:
            pass
        else:
            raise Exception('Invalid device specified, must be in the form [ "cpu:idx" | "gpu:idx" ]')
    return x


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev=None):
    if dtype_str:
        dtype = _np.__dict__[dtype_str]
    else:
        dtype = None
    return _to_dev(_np.array(object_in, dtype=dtype), dev)


to_numpy = lambda x: x
to_list = lambda x: x.tolist()
shape = lambda x: x.shape
get_num_dims = lambda x: len(x.shape)
minimum = _np.minimum
maximum = _np.maximum
clip = _np.clip
round = _np.round
floormod = lambda x, y: x % y
floor = _np.floor
ceil = _np.ceil
abs = _np.absolute
argmax = _np.argmax
argmin = _np.argmin


def cast(x, dtype_str):
    dtype_val = _np.__dict__[dtype_str]
    return x.astype(dtype_val)


# noinspection PyShadowingNames
def arange(stop, start=0, step=1, dtype_str=None, dev=None):
    if dtype_str:
        dtype = _np.__dict__[dtype_str]
    else:
        dtype = None
    res = _to_dev(_np.arange(start, stop, step=step, dtype=dtype), dev)
    if not dtype:
        if res.dtype == _np.dtype('float64'):
            return res.astype(_np.float32)
        elif res.dtype == _np.dtype('int64'):
            return res.astype(_np.int32)
    return res


def linspace(start, stop, num, axis=None, dev=None):
    if axis is None:
        axis = -1
    return _to_dev(_np.linspace(start, stop, num, axis=axis), dev)


def concatenate(xs, axis=None):
    if axis is None:
        xs = [reshape(a, (-1,)) for a in xs]
        axis = 0
    return _np.concatenate(xs, axis)


def flip(x, axis=None, batch_shape=None):
    num_dims = len(batch_shape) if batch_shape is not None else len(x.shape)
    if axis is None:
        axis = list(range(num_dims))
    if type(axis) is int:
        axis = [axis]
    axis = [item + num_dims if item < 0 else item for item in axis]
    return _np.flip(x, axis)


stack = _np.stack


def unstack(x, axis, _=None):
    x_split = _np.split(x, x.shape[axis], axis)
    res = [_np.squeeze(item, axis) for item in x_split]
    return res


def split(x, num_sections=None, axis=0):
    dim_size = x.shape[axis]
    if num_sections is None:
        num_sections = dim_size
    return _np.split(x, num_sections, axis)


tile = _np.tile
constant_pad = lambda x, pad_width, value=0, _=None: _np.pad(x, pad_width, constant_values=value)
zero_pad = lambda x, pad_width, _=None: _np.pad(x, pad_width)
swapaxes = _np.swapaxes


def transpose(x, axes=None):
    if axes is None:
        num_dims = len(x.shape)
        axes = list(range(num_dims))
        axes.reverse()
    return _np.transpose(x, axes)


expand_dims = _np.expand_dims
where = lambda condition, x1, x2, _=None, _1=None: _np.where(condition, x1, x2)


def indices_where(x):
    where_x = _np.where(x)
    res = _np.concatenate([_np.expand_dims(item, -1) for item in where_x], -1)
    return res


reshape = _np.reshape
squeeze = _np.squeeze


# noinspection PyShadowingNames
def zeros(shape, dtype_str='float32', dev=None):
    dtype = _np.__dict__[dtype_str]
    return _to_dev(_np.zeros(shape, dtype), dev)


# noinspection PyShadowingNames
def zeros_like(x, dtype_str=None, dev=None):
    if dtype_str:
        dtype = _np.__dict__[dtype_str]
    else:
        dtype = x.dtype
    return _to_dev(_np.zeros_like(x, dtype=dtype), dev)


# noinspection PyShadowingNames
def ones(shape, dtype_str='float32', dev=None):
    dtype = _np.__dict__[dtype_str]
    return _to_dev(_np.ones(shape, dtype), dev)


# noinspection PyShadowingNames
def ones_like(x, dtype_str=None, dev=None):
    if dtype_str:
        dtype = _np.__dict__[dtype_str]
    else:
        dtype = x.dtype
    return _to_dev(_np.ones_like(x, dtype=dtype), dev)


# noinspection PyUnusedLocal
def one_hot(indices, depth, dev=None):
    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    res = _np.eye(depth)[_np.array(indices).reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


cross = _np.cross
matmul = lambda x1, x2, _=None: _np.matmul(x1, x2)
cumsum = _np.cumsum


# noinspection PyShadowingNames
def identity(n, dtype_str='float32', batch_shape=None, dev=None):
    dtype = _np.__dict__[dtype_str]
    mat = _np.identity(n, dtype=dtype)
    if batch_shape is None:
        return_mat = mat
    else:
        reshape_dims = [1] * len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        return_mat = _np.tile(_np.reshape(mat, reshape_dims), tile_dims)
    return _to_dev(return_mat, dev)


def scatter_flat(indices, updates, size, reduction='sum', dev=None):
    if dev is None:
        dev = dev_str(updates)
    if reduction == 'sum':
        target = _np.zeros([size], dtype=updates.dtype)
        _np.add.at(target, indices, updates)
    elif reduction == 'min':
        target = _np.ones([size], dtype=updates.dtype) * 1e12
        _np.minimum.at(target, indices, updates)
        target = _np.where(target == 1e12, 0., target)
    elif reduction == 'max':
        target = _np.ones([size], dtype=updates.dtype) * -1e12
        _np.maximum.at(target, indices, updates)
        target = _np.where(target == -1e12, 0., target)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    return _to_dev(target, dev)


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape, num_idx_dims=None, reduction='sum', dev=None):
    if dev is None:
        dev = dev_str(updates)
    shape = list(shape)
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    if reduction == 'sum':
        target = _np.zeros(shape, dtype=updates.dtype)
        _np.add.at(target, indices_tuple, updates)
    elif reduction == 'min':
        target = _np.ones(shape, dtype=updates.dtype) * 1e12
        _np.minimum.at(target, indices_tuple, updates)
        target = _np.where(target == 1e12, 0., target)
    elif reduction == 'max':
        target = _np.ones(shape, dtype=updates.dtype) * -1e12
        _np.maximum.at(target, indices_tuple, updates)
        target = _np.where(target == -1e12, 0., target)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    return _to_dev(target, dev)


def gather_flat(params, indices, dev=None):
    if dev is None:
        dev = dev_str(params)
    return _to_dev(_np.take(params, indices, 0), dev)


def gather_nd(params, indices, indices_shape=None, dev=None):
    if dev is None:
        dev = dev_str(params)
    if indices_shape is None:
        indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(_mul, params_shape[i + 1:], 1) for i in range(len(params_shape) - 1)] + [1]
    result_dim_sizes = _np.array(result_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = _np.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = _np.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = _np.tile(_np.reshape(_np.sum(indices * indices_scales, -1, keepdims=True), (-1, 1)), (1, implicit_indices_factor))
    implicit_indices = _np.tile(_np.expand_dims(_np.arange(implicit_indices_factor), 0), (indices_for_flat_tiled.shape[0], 1))
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = _np.reshape(indices_for_flat, (-1,)).astype(_np.int32)
    flat_gather = _np.take(flat_params, flat_indices_for_flat, 0)
    new_shape = list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    res = _np.reshape(flat_gather, new_shape)
    return _to_dev(res, dev)


dev = lambda x: 'cpu:0'
dev_str = lambda x: 'cpu:0'
dev_to_str = lambda dev_in: 'cpu:0'
gpu_is_available = lambda: False
tpu_is_available = lambda: False
dtype = lambda x: x.dtype
dtype_str = lambda x: DTYPE_DICT[x.dtype]
dtype_to_str = lambda dtype_in: DTYPE_DICT[dtype_in]


# noinspection PyUnusedLocal
def compile_fn(func, dynamic=True, example_inputs=None):
    logging.warning('Numpy does not support compiling functions.\n'
                    'Now returning the unmodified function.')
    return func
