"""
Collection of TensorFlow general functions, wrapped to fit Ivy syntax and signature.
"""

# global
from functools import reduce as _reduce
from operator import mul as _mul

_round = round

# global
import tensorflow as _tf
import numpy as _np

DTYPE_DICT = {_tf.int32: 'int32',
              _tf.int64: 'int64',
              _tf.float32: 'float32',
              _tf.float64: 'float64'}


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev=None):
    dtype = _tf.__dict__[dtype_str] if dtype_str else dtype_str
    if dev:
        with _tf.device('/' + dev.upper()):
            return _tf.convert_to_tensor(object_in, dtype=dtype)
    else:
        return _tf.convert_to_tensor(object_in, dtype=dtype)


to_numpy = lambda x: _np.asarray(_tf.convert_to_tensor(x))
to_list = lambda x: x.numpy().tolist()
shape = _tf.shape
get_num_dims = lambda x: _tf.shape(_tf.shape(x))
minimum = _tf.minimum
maximum = _tf.maximum
clip = _tf.clip_by_value
# noinspection PyShadowingBuiltins
round = _tf.round
floormod = lambda x, y: x % y
floor = _tf.floor
ceil = _tf.math.ceil
# noinspection PyShadowingBuiltins
abs = _tf.math.abs
argmax = _tf.math.argmax
argmin = _tf.math.argmin


def cast(x, dtype_str):
    dtype_val = _tf.__dict__[dtype_str]
    return _tf.cast(x, dtype_val)


# noinspection PyShadowingNames
def arange(stop, start=0, step=1, dtype_str=None, dev=None):
    dtype = _tf.__dict__[dtype_str] if dtype_str else dtype_str
    if dev:
        with _tf.device('/' + dev.upper()):
            return _tf.range(start, stop, delta=step, dtype=dtype)
    else:
        return _tf.range(start, stop, delta=step, dtype=dtype)


def linspace(start, stop, num, axis=None, dev=None):
    if axis is None:
        axis = -1
    if dev:
        with _tf.device('/' + dev.upper()):
            return _tf.linspace(start, stop, num, axis=axis)
    else:
        return _tf.linspace(start, stop, num, axis=axis)


def concatenate(xs, axis=None):
    if axis is None:
        xs = [_tf.reshape(a, (-1,)) for a in xs]
        axis = 0
    return _tf.concat(xs, axis)


def flip(x, axis=None, batch_shape=None):
    num_dims = len(batch_shape) if batch_shape is not None else len(x.shape)
    if axis is None:
        new_axis = list(range(num_dims))
    else:
        new_axis = axis
    if type(new_axis) is int:
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return _tf.reverse(x, new_axis)


stack = _tf.stack
unstack = lambda x, axis, _=None: _tf.unstack(x, axis=axis)


def split(x, num_sections=None, axis=0):
    if num_sections is None:
        dim_size = _tf.shape(x)[axis]
        num_sections = dim_size
    return _tf.split(x, num_sections, axis)


tile = _tf.tile
constant_pad = lambda x, pad_width, value=0, _=None: _tf.pad(x, pad_width, constant_values=value)
zero_pad = lambda x, pad_width, _=None: _tf.pad(x, pad_width)


def swapaxes(x, axis0, axis1):
    x_shape = x.shape
    num_dims = len(x_shape)
    axis0 %= num_dims
    axis1 %= num_dims
    config = list(range(num_dims))
    config.pop(axis0)
    config.insert(axis0, axis1)
    config.pop(axis1)
    config.insert(axis1, axis0)
    return _tf.transpose(x, config)


transpose = _tf.transpose
expand_dims = _tf.expand_dims
where = lambda condition, x1, x2, _=None, _1=None: _tf.where(condition, x1, x2)
indices_where = _tf.where
reshape = _tf.reshape
squeeze = _tf.squeeze


# noinspection PyShadowingNames
def zeros(shape, dtype_str='float32', dev=None):
    dtype = _tf.__dict__[dtype_str]
    if dev:
        with _tf.device('/' + dev.upper()):
            return _tf.zeros(shape, dtype)
    else:
        return _tf.zeros(shape, dtype)


# noinspection PyShadowingNames
def zeros_like(x, dtype_str=None, dev=None):
    dtype = _tf.__dict__[dtype_str] if dtype_str else dtype_str
    if dev:
        with _tf.device('/' + dev.upper()):
            return _tf.zeros_like(x, dtype=dtype)
    else:
        return _tf.zeros_like(x, dtype=dtype)


# noinspection PyShadowingNames
def ones(shape, dtype_str='float32', dev=None):
    dtype = _tf.__dict__[dtype_str]
    if dev:
        with _tf.device('/' + dev.upper()):
            return _tf.ones(shape, dtype)
    else:
        return _tf.ones(shape, dtype)


# noinspection PyShadowingNames
def ones_like(x, dtype_str=None, dev=None):
    dtype = _tf.__dict__[dtype_str] if dtype_str else dtype_str
    if dev:
        with _tf.device('/' + dev.upper()):
            return _tf.ones_like(x, dtype=dtype)
    else:
        return _tf.ones_like(x, dtype=dtype)


def one_hot(indices, depth, dev=None):
    if dev is not None:
        with _tf.device('/' + dev.upper()):
            return _tf.one_hot(indices, depth)
    return _tf.one_hot(indices, depth)


cross = _tf.linalg.cross
matmul = lambda x1, x2, _=None: _tf.matmul(x1, x2)
cumsum = _tf.cumsum


# noinspection PyShadowingNames
def identity(n, dtype_str='float32', batch_shape=None, dev=None):
    dtype = _tf.__dict__[dtype_str]
    if dev:
        with _tf.device('/' + dev.upper()):
            return _tf.eye(n, n, batch_shape=batch_shape, dtype=dtype)
    else:
        return _tf.eye(n, n, batch_shape=batch_shape, dtype=dtype)


TF_SCATTER_VAR = {}


# noinspection PyShadowingNames
def scatter_flat(indices, updates, size, reduction='sum', dev=None):
    if dev is None:
        dev = dev_str(updates)
    dtype = updates.dtype
    if reduction == 'sum':
        return _tf.scatter_nd(_tf.expand_dims(indices, -1), updates, [size])
    elif reduction == 'min':
        func = _tf.compat.v1.scatter_min
        initial_val = _tf.cast(_tf.constant(2 ** 31 - 1), dtype)
    elif reduction == 'max':
        func = _tf.compat.v1.scatter_max
        initial_val = _tf.cast(_tf.constant(-(2 ** 31 - 1)), dtype)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    global TF_SCATTER_VAR
    if size not in TF_SCATTER_VAR:
        TF_SCATTER_VAR[size] = {dtype: _tf.Variable(_tf.ones(size, dtype=dtype) * initial_val, trainable=False)}
    elif dtype not in TF_SCATTER_VAR[size]:
        TF_SCATTER_VAR[size][dtype] = _tf.Variable(_tf.ones(size, dtype=dtype) * initial_val, trainable=False)
    else:
        TF_SCATTER_VAR[size][dtype].assign(_tf.ones(size, dtype=dtype) * initial_val)
    res = _tf.convert_to_tensor(func(TF_SCATTER_VAR[size][dtype], indices, updates))
    res = _tf.where(res == initial_val, _tf.zeros(size, dtype=updates.dtype), res)
    with _tf.device('/' + dev.upper()):
        return res


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape, num_idx_dims=None, reduction='sum', dev=None):
    if dev is None:
        dev = dev_str(updates)
    shape = list(shape)
    dtype = updates.dtype
    if reduction == 'sum':
        return _tf.scatter_nd(indices, updates, shape)
    elif reduction == 'min':
        func = _tf.compat.v1.scatter_min
        initial_val = _tf.cast(_tf.constant(2 ** 31 - 1), dtype)
    elif reduction == 'max':
        func = _tf.compat.v1.scatter_max
        initial_val = _tf.cast(_tf.constant(-(2 ** 31 - 1)), dtype)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    indices_shape = indices.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(_mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [1]
    result_dim_sizes = _tf.constant(result_dim_sizes_list)
    implicit_indices_factor = result_dim_sizes[num_index_dims - 1]
    flat_result_size = _reduce(_mul, shape, 1)
    global TF_SCATTER_VAR
    if flat_result_size not in TF_SCATTER_VAR:
        TF_SCATTER_VAR[flat_result_size] = {dtype: _tf.Variable(_tf.ones(flat_result_size, dtype=dtype) * initial_val, trainable=False)}
    elif dtype not in TF_SCATTER_VAR[flat_result_size]:
        TF_SCATTER_VAR[flat_result_size][dtype] = _tf.Variable(_tf.ones(flat_result_size, dtype=dtype) * initial_val, trainable=False)
    else:
        TF_SCATTER_VAR[flat_result_size][dtype].assign(_tf.ones(flat_result_size, dtype=dtype) * initial_val)
    flat_updates = _tf.reshape(updates, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = _tf.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = _tf.tile(_tf.reshape(_tf.reduce_sum(indices * indices_scales, -1, keepdims=True), (-1, 1)), [1, implicit_indices_factor])
    implicit_indices = _tf.tile(_tf.expand_dims(_tf.range(implicit_indices_factor), 0), _tf.stack((_tf.shape(indices_for_flat_tiled)[0], _tf.constant(1))))
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = _tf.reshape(indices_for_flat, (-1,))
    flat_scatter = _tf.convert_to_tensor(func(TF_SCATTER_VAR[flat_result_size][dtype], flat_indices_for_flat, flat_updates))
    flat_scatter = _tf.where(flat_scatter == initial_val, _tf.zeros(flat_result_size, dtype=updates.dtype), flat_scatter)
    with _tf.device('/' + dev.upper()):
        res = _tf.reshape(flat_scatter, list(shape))
        return res


def gather_flat(params, indices, dev=None):
    if dev is None:
        dev = dev_str(params)
    with _tf.device('/' + dev.upper()):
        return _tf.gather_nd(params, _tf.expand_dims(indices, -1))


def gather_nd(params, indices, _=None, dev=None):
    if dev is None:
        dev = dev_str(params)
    with _tf.device('/' + dev.upper()):
        return _tf.gather_nd(params, indices)


dev = lambda x: x.device


def dev_to_str(dev_in):
    return ':'.join(dev_in.split(':')[-2:])


dev_str = lambda x: dev_to_str(dev(x))
dtype = lambda x: x.dtype
dtype_str = lambda x: DTYPE_DICT[x.dtype]
dtype_to_str = lambda dtype_in: DTYPE_DICT[dtype_in]
compile_fn = lambda fn, dynamic=True, example_inputs=None: _tf.function(fn)
