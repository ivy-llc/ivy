"""
Collection of MXNet general functions, wrapped to fit Ivy syntax and signature.
"""

# global
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
from ivy.core.device import default_device
from ivy.mxnet.core.device import _callable_dev_str


DTYPE_DICT = {_np.bool_: 'bool',
              _np.int8: 'int8',
              _np.uint8: 'uint8',
              _np.int16: 'int16',
              _np.int32: 'int32',
              _np.int64: 'int64',
              _np.float16: 'float16',
              _np.float32: 'float32',
              _np.float64: 'float64'}


# Helpers #
# --------#

def _raise(ex):
    raise ex


def _mxnet_init_context(dev):
    if dev is None or dev.find("cpu") != -1:
        mx_dev = "cpu"
    elif dev.find("gpu") != -1:
        mx_dev = "gpu"
    else:
        raise Exception("dev input {} not supported.".format(dev))
    if dev.find(":") != -1:
        mx_dev_id = int(dev[dev.find(":")+1:])
    else:
        mx_dev_id = 0
    return _mx.Context(mx_dev, mx_dev_id)


def _scalar_or_flat_array_to_scalar(x):
    return x if isinstance(x, Number) else (x.asscalar() if x.shape == () else x)


def _flat_array_to_1_dim_array(x):
    return _mx.nd.array([x.asscalar()]) if x.shape == () else x


def _1_dim_array_to_flat_array(x):
    return _mx.nd.array(x.asscalar(), dtype=x.dtype) if x.shape == (1,) else x


# API #
# ----#

def array(object_in, dtype_str=None, dev_str=None):
    cont = _mxnet_init_context(default_device(dev_str))
    return _mx.nd.array(object_in, cont, dtype=dtype_str)


def is_array(x, exclusive=False):
    if isinstance(x, _mx.ndarray.ndarray.NDArray):
        if exclusive and x.grad is not None:
            return False
        return True
    return False


copy_array = lambda x: x.copy()
array_equal = lambda x0, x1: _mx.nd.min(_mx.nd.broadcast_equal(x0, x1)) == 1
to_numpy = lambda x: x if isinstance(x, _np.ndarray) else (_np.array(x) if isinstance(x, (int, float)) else x.asnumpy())
to_numpy.__name__ = 'to_numpy'
to_scalar = lambda x: x.asscalar().item()
to_scalar.__name__ = 'to_scalar'
to_list = lambda x: x.asnumpy().tolist()
to_list.__name__ = 'to_list'
shape = lambda x, as_tensor=False: _mx.nd.shape_array(x) if as_tensor else x.shape
shape.__name__ = 'shape'
get_num_dims = lambda x, as_tensor=False:\
    _mx.nd.shape_array(_mx.nd.shape_array(x)).reshape([]) if as_tensor else len(x.shape)
minimum = lambda x, y: _mx.nd.array(_mx.nd.minimum(_scalar_or_flat_array_to_scalar(x), _scalar_or_flat_array_to_scalar(y)))
maximum = lambda x, y: _mx.nd.array(_mx.nd.maximum(_scalar_or_flat_array_to_scalar(x), _scalar_or_flat_array_to_scalar(y)))
clip = lambda x, x_min, x_max: _mx.nd.clip(_flat_array_to_1_dim_array(_mx.nd.array(x)),
                                           _scalar_or_flat_array_to_scalar(x_min),
                                           _scalar_or_flat_array_to_scalar(x_max))


def round(x):
    if len(x.shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.round(_flat_array_to_1_dim_array(x)))
    return _mx.nd.round(x)


def floormod(x, y):
    orig_x_shape = x.shape
    if len(orig_x_shape) == 0:
        x = _flat_array_to_1_dim_array(x)
    if len(y.shape) == 0:
        y = _flat_array_to_1_dim_array(y)
    res = x % y
    if len(orig_x_shape) == 0:
        return _1_dim_array_to_flat_array(res)
    return res


def floor(x):
    if len(x.shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.floor(_flat_array_to_1_dim_array(x)))
    return _mx.nd.floor(x)


def ceil(x):
    if len(x.shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.ceil(_flat_array_to_1_dim_array(x)))
    return _mx.nd.ceil(x)


# noinspection PyShadowingBuiltins
def abs(x):
    if len(x.shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.abs(_flat_array_to_1_dim_array(x)))
    return _mx.nd.abs(x)


argmax = lambda x, axis=0: _mx.nd.argmax(x, axis)
argmin = lambda x, axis=0: _mx.nd.argmin(x, axis)
argsort = lambda x, axis=-1: _mx.nd.argsort(x, axis)
cast = lambda x, dtype_str: x.astype(dtype_str)


# noinspection PyUnresolvedReferences
def arange(stop, start=0, step=1, dtype_str=None, dev_str=None):
    cont = _mxnet_init_context(default_device(dev_str))
    stop = stop if isinstance(stop, Number) else stop.asscalar()
    start = start if isinstance(start, Number) else start.asscalar()
    step = step if isinstance(step, Number) else step.asscalar()
    return _mx.nd.arange(start, stop, ctx=cont, step=step, dtype=dtype_str)


def _linspace(start, stop, num, cont):
    if num == 1:
        return start
    start = _mx.nd.array(start).reshape((1,)).astype('float32')
    stop = _mx.nd.array(stop).reshape((1,)).astype('float32')
    n_m_1 = _mx.nd.array(num - 1).reshape((1,)).astype('float32')
    increment = (stop - start)/n_m_1
    increment_tiled = _mx.nd.tile(increment, num - 1)
    increments = increment_tiled * _mx.nd.array(_mx.nd.np.linspace(1, num - 1, num - 1).tolist(), ctx=cont)
    ret = _mx.nd.concat(start, start + increments, dim=0)
    return ret


def linspace(start, stop, num, axis=None, dev_str=None):
    cont = _mxnet_init_context(default_device(dev_str))
    num = num.asnumpy()[0] if isinstance(num, _mx.nd.NDArray) else num
    start_is_array = isinstance(start, _mx.nd.NDArray)
    stop_is_array = isinstance(stop, _mx.nd.NDArray)
    start_shape = []
    if start_is_array:
        start_shape = list(start.shape)
        start = start.reshape((-1,))
    if stop_is_array:
        start_shape = list(stop.shape)
        stop = stop.reshape((-1,))
    if start_is_array and stop_is_array:
        res = [_linspace(strt, stp, num, cont) for strt, stp in zip(start, stop)]
    elif start_is_array and not stop_is_array:
        res = [_linspace(strt, stop, num, cont) for strt in start]
    elif not start_is_array and stop_is_array:
        res = [_linspace(start, stp, num, cont) for stp in stop]
    else:
        return _linspace(start, stop, num, cont)
    new_shape = start_shape + [num]
    res = _mx.nd.concat(*res, dim=-1).reshape(new_shape)
    if axis is not None:
        res = _mx.nd.swapaxes(res, axis, -1)
    return res


def logspace(start, stop, num, base=10., axis=None, dev_str=None):
    power_seq = linspace(start, stop, num, axis, default_device(dev_str))
    return base ** power_seq


def concatenate(xs, axis=-1):
    if xs[0].shape == ():
        return _mx.nd.concat(*[_flat_array_to_1_dim_array(x) for x in xs], dim=axis)
    return _mx.nd.concat(*xs, dim=axis)


def flip(x, axis=None, batch_shape=None):
    num_dims = len(batch_shape) if batch_shape is not None else len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        new_axis = list(range(num_dims))
    else:
        new_axis = axis
    if type(new_axis) is int:
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return _mx.nd.flip(x, new_axis)


def stack(xs, axis=0):
    if xs[0].shape == ():
        return _mx.nd.reshape(_mx.nd.stack(*[_flat_array_to_1_dim_array(x) for x in xs], axis=axis), -1)
    return _mx.nd.stack(*xs, axis=axis)


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    num_outputs = x.shape[axis]
    ret = _mx.nd.split(x, num_outputs, axis, squeeze_axis=not keepdims)
    return ret if isinstance(ret, list) else [ret]


def split(x, num_or_size_splits=None, axis=0, with_remainder=False):
    if isinstance(num_or_size_splits, (list, tuple)):
        raise Exception('MXNet does not support splitting based on section sizes,'
                        'only number of sections as integer input is supported.')
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception('input array had no shape, but num_sections specified was {}'.format(num_or_size_splits))
        return [x]
    if num_or_size_splits == 1:
        return [x]
    elif with_remainder:
        num_or_size_splits = x.shape[axis] if not num_or_size_splits else num_or_size_splits
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = _math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            raise Exception('MXNet does not support splitting along an axis with a remainder after division.')
    return _mx.nd.split(x, x.shape[axis] if not num_or_size_splits else num_or_size_splits, axis)


def repeat(x, repeats, axis=None):
    return _mx.nd.repeat(_flat_array_to_1_dim_array(x), repeats, axis)


def tile(x, reps):
    if isinstance(reps, _mx.nd.ndarray.NDArray):
        reps = reps.asnumpy().tolist()
    return _mx.nd.tile(_flat_array_to_1_dim_array(x), reps)


def constant_pad(x, pad_width, value=0):
    x = _flat_array_to_1_dim_array(x)
    if isinstance(pad_width, _mx.ndarray.ndarray.NDArray):
        pad_width = pad_width.asnumpy().tolist()
    x_shape = list(x.shape)
    num_dims = len(x_shape)
    if num_dims > 3:
        raise Exception('Invalid inputs. Pad for mxnet only supports inputs with 3 dimensions or smaller.')
    num_dims_to_add = 4 - num_dims
    new_shape = tuple([1] * num_dims_to_add + x_shape)
    mat_expanded_dims = _mx.nd.reshape(x, new_shape)
    pad_width_flat = [0]*num_dims_to_add*2 + [item for sublist in pad_width for item in sublist]
    pad_expanded_dims = _mx.nd.pad(mat_expanded_dims, mode="constant", pad_width=tuple(pad_width_flat),
                                   constant_value=value)
    new_shape = [orig_dim + pad_width_item[0] + pad_width_item[1] for orig_dim, pad_width_item in zip(x_shape, pad_width)]
    res = _mx.nd.reshape(pad_expanded_dims, tuple(new_shape))
    return res


def zero_pad(x, pad_width):
    return constant_pad(x, pad_width, 0)


swapaxes = _mx.nd.swapaxes


def transpose(x, axes=None):
    if axes is None:
        num_dims = len(x.shape)
        axes = list(range(num_dims))
        axes.reverse()
    return _mx.nd.transpose(x, axes)


def expand_dims(x, axis):
    if x.shape == ():
        return _flat_array_to_1_dim_array(x)
    return _mx.nd.expand_dims(x, axis)


def where(condition, x1, x2):
    flat = False
    if condition.shape == ():
        condition = _flat_array_to_1_dim_array(condition)
        flat = True
    if x1.shape == ():
        x1 = _flat_array_to_1_dim_array(x1)
        flat = True
    if x2.shape == ():
        x2 = _flat_array_to_1_dim_array(x2)
        flat = True
    x_shape = list(x1.shape)
    condition_shape = list(condition.shape)
    if x_shape == condition_shape:
        res = _mx.nd.where(condition, x1, x2)
        if flat:
            return _1_dim_array_to_flat_array(res)
        return res
    tile_reps = [int(x / c) for x, c in zip(x_shape, condition_shape)]
    tiled_condition = _mx.nd.tile(condition, tile_reps)
    res = _mx.nd.where(tiled_condition, x1, x2)
    if flat:
        return _1_dim_array_to_flat_array(res)
    return res


def indices_where(x):
    x_shape = x.shape
    x_flat = x.reshape((1, -1,))
    flat_indices = x_flat.astype('int32').tostype('csr').indices
    if flat_indices.shape == (0,):
        res = flat_indices.reshape((0, len(x_shape)))
        return res
    res = _mx.nd.swapaxes(_mx.nd.unravel_index(flat_indices, x_shape), 0, 1)
    return res


isnan = _mx.nd.contrib.isnan
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
def zeros(shape, dtype_str='float32', dev_str=None):
    cont = _mxnet_init_context(default_device(dev_str))
    if len(shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.zeros((1,), ctx=cont).astype(dtype_str))
    return _mx.nd.zeros(shape, ctx=cont).astype(dtype_str)


def zeros_like(x, dtype_str=None, dev_str=None):
    if x.shape == ():
        return _mx.nd.array(0., ctx=_mxnet_init_context(default_device(dev_str)))
    mx_zeros = _mx.nd.zeros_like(x, ctx=_mxnet_init_context(default_device(dev_str)))
    return mx_zeros if not dtype_str else mx_zeros.astype(dtype_str)


# noinspection PyShadowingNames
def ones(shape, dtype_str='float32', dev_str=None):
    cont = _mxnet_init_context(default_device(dev_str))
    if len(shape) == 0:
        return _1_dim_array_to_flat_array(_mx.nd.ones((1,), ctx=cont).astype(dtype_str))
    return _mx.nd.ones(shape, ctx=cont).astype(dtype_str)


def ones_like(x, dtype_str=None, dev_str=None):
    if x.shape == ():
        return _mx.nd.array(1., ctx=_mxnet_init_context(default_device(dev_str)))
    mx_ones = _mx.nd.ones_like(x, ctx=_mxnet_init_context(default_device(dev_str)))
    return mx_ones if dtype_str is None else mx_ones.astype(dtype_str)


# noinspection PyUnusedLocal
one_hot = lambda indices, depth, dev_str=None: _mx.nd.one_hot(indices, depth)


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


cumsum = lambda x, axis=0: _mx.nd.cumsum(x, axis if axis >= 0 else axis % len(x.shape))


def cumprod(x, axis=0, exclusive=False):
    array_stack = [_mx.nd.expand_dims(chunk, axis) for chunk in unstack(x, axis)]
    if exclusive:
        array_stack = [_mx.nd.ones_like(array_stack[0])] + array_stack[:-1]
    new_array_list = [array_stack[0]]
    for array_chunk in array_stack[1:]:
        new_array_list.append(new_array_list[-1] * array_chunk)
    return _mx.nd.concat(*new_array_list, dim=axis)


def identity(n, dtype_str='float32', batch_shape=None, dev_str=None):
    mat = _mx.nd.eye(n, dtype=dtype_str).copyto(_mxnet_init_context(default_device(dev_str)))
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


# noinspection PyShadowingNames
def scatter_flat(indices, updates, size, reduction='sum', dev_str=None):
    if reduction == 'replace':
        return _mx.nd.scatter_nd(updates, _mx.nd.expand_dims(indices, 0), [size]).copyto(_mxnet_init_context(default_device(dev_str)))
    else:
        raise Exception('MXNet scatter_flat currently only supports reduction mode "replace", but {} selected.'.
                        format(reduction))


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape, reduction='sum', dev_str=None):
    if dev_str is None:
        dev_str = _callable_dev_str(indices)
    shape = list(shape)
    num_idx_dims = len(indices.shape)
    transpose_order = [num_idx_dims-1] + list(range(num_idx_dims-1))
    indices = _mx.nd.transpose(indices, transpose_order)
    shape = shape if type(shape) is list else shape.asnumpy().astype(_np.int32).tolist()
    if reduction == 'replace':
        return _mx.nd.scatter_nd(updates, indices, shape).copyto(_mxnet_init_context(dev_str))
    else:
        raise Exception('MXNet scatter_nd currently only supports reduction mode "replace", but {} selected.'.
                        format(reduction))


def gather(params, indices, axis=-1, dev_str=None):
    if dev_str is None:
        dev_str = _callable_dev_str(params)
    index_slices = unstack(indices, -1)
    res = _mx.nd.concat(
        *[_mx.nd.expand_dims(_mx.nd.pick(params, idx_slice, axis), -1) for idx_slice in index_slices], dim=-1)
    res = _mx.nd.reshape(res, indices.shape)
    return res.copyto(_mxnet_init_context(dev_str))


def gather_nd(params, indices, dev_str=None):
    if dev_str is None:
        dev_str = _callable_dev_str(params)
    indices_shape = indices.shape
    num_idx_dims = len(indices_shape)
    transpose_order = [num_idx_dims-1] + list(range(num_idx_dims-1))
    indices = _mx.nd.transpose(indices, transpose_order)
    return _mx.nd.gather_nd(params, indices).copyto(_mxnet_init_context(dev_str))


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
    xp = _mx.nd.reshape(_mx.nd.arange(num_vals*x_pre_size*x_post_size), x_shape)
    x_coords = _mx.nd.arange(num_samples) * ((num_vals-1)/(num_samples-1)) * x_post_size
    x_coords = _mx.nd.reshape(x_coords, [1]*num_pre_dims + [num_samples] + [1]*num_post_dims)
    x_coords = _mx.nd.broadcast_to(x_coords, x_pre_shape + [num_samples] + x_post_shape)
    slc = [slice(None)] * num_x_dims
    slc[axis] = slice(0, 1, 1)
    x_coords = x_coords + xp[tuple(slc)]
    x = _mx.nd.reshape(x, (-1,))
    xp = _mx.nd.reshape(xp, (-1,))
    x_coords = _mx.nd.reshape(x_coords, (-1,))
    ret = _mx.nd.array(_mx.np.interp(x_coords.asnumpy(), xp.asnumpy(), x.asnumpy()))
    return _mx.nd.reshape(ret, x_pre_shape + [num_samples] + x_post_shape)


dtype = lambda x: x.dtype
dtype.__name__ = 'dtype'
dtype_str = lambda x: DTYPE_DICT[x.dtype]
dtype_to_str = lambda dtype_in: DTYPE_DICT[dtype_in]


# noinspection PyUnusedLocal
def compile(func, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None):
    logging.warning('MXnet does not support compiling arbitrary functions, '
                    'consider writing a function using MXNet Symbolic backend instead for compiling.\n'
                    'Now returning the unmodified function.')
    return func


current_framework_str = lambda: 'mxnet'
current_framework_str.__name__ = 'current_framework_str'
multiprocessing = lambda context=None: _multiprocessing if context is None else _multiprocessing.get_context(context)
container_types = lambda: []
