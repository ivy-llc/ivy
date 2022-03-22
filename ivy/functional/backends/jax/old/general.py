"""
Collection of Jax general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax as _jax
import math as _math
import numpy as _onp
import jax.numpy as _jnp
import jaxlib as _jaxlib
from numbers import Number
from collections import Iterable
from operator import mul as _mul
from functools import reduce as _reduce
from jaxlib.xla_extension import Buffer
import multiprocessing as _multiprocessing
from haiku._src.data_structures import FlatMapping

# local
import ivy
from ivy.functional.ivy.device import default_device
from ivy.functional.ivy import default_dtype
from ivy.functional.backends.jax.device import to_dev, dev as callable_dev

DTYPE_TO_STR = {_jnp.dtype('int8'): 'int8',
                _jnp.dtype('int16'): 'int16',
                _jnp.dtype('int32'): 'int32',
                _jnp.dtype('int64'): 'int64',
                _jnp.dtype('uint8'): 'uint8',
                _jnp.dtype('uint16'): 'uint16',
                _jnp.dtype('uint32'): 'uint32',
                _jnp.dtype('uint64'): 'uint64',
                _jnp.dtype('bfloat16'): 'bfloat16',
                _jnp.dtype('float16'): 'float16',
                _jnp.dtype('float32'): 'float32',
                _jnp.dtype('float64'): 'float64',
                _jnp.dtype('bool'): 'bool',

                _jnp.int8: 'int8',
                _jnp.int16: 'int16',
                _jnp.int32: 'int32',
                _jnp.int64: 'int64',
                _jnp.uint8: 'uint8',
                _jnp.uint16: 'uint16',
                _jnp.uint32: 'uint32',
                _jnp.uint64: 'uint64',
                _jnp.bfloat16: 'bfloat16',
                _jnp.float16: 'float16',
                _jnp.float32: 'float32',
                _jnp.float64: 'float64',
                _jnp.bool_: 'bool'}

DTYPE_FROM_STR = {'int8': _jnp.dtype('int8'),
                  'int16': _jnp.dtype('int16'),
                  'int32': _jnp.dtype('int32'),
                  'int64': _jnp.dtype('int64'),
                  'uint8': _jnp.dtype('uint8'),
                  'uint16': _jnp.dtype('uint16'),
                  'uint32': _jnp.dtype('uint32'),
                  'uint64': _jnp.dtype('uint64'),
                  'bfloat16': _jnp.dtype('bfloat16'),
                  'float16': _jnp.dtype('float16'),
                  'float32': _jnp.dtype('float32'),
                  'float64': _jnp.dtype('float64'),
                  'bool': _jnp.dtype('bool')}


# Helpers #
# --------#



def _to_array(x):
    if isinstance(x, _jax.interpreters.ad.JVPTracer):
        return _to_array(x.primal)
    elif isinstance(x, _jax.interpreters.partial_eval.DynamicJaxprTracer):
        return _to_array(x.aval)
    return x


# API #
# ----#











def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace('uint', '').replace('int', '').replace('bfloat', '').replace('float', ''))



shape = lambda x, as_tensor=False: _jnp.asarray(_jnp.shape(x)) if as_tensor else x.shape
shape.__name__ = 'shape'
get_num_dims = lambda x, as_tensor=False: _jnp.asarray(len(_jnp.shape(x))) if as_tensor else len(x.shape)
minimum = _jnp.minimum
maximum = _jnp.maximum
clip = _jnp.clip
# noinspection PyShadowingBuiltins
# noinspection PyShadowingBuiltins
abs = _jnp.absolute

def argmin(x, axis=0):
    ret = _jnp.argmin(x, axis)
    if ret.shape == ():
        return ret.reshape(-1)
    return ret


def cast(x, dtype):
    return x.astype(dtype_from_str(dtype))


astype = cast


# noinspection PyShadowingNames
def arange(stop, start=0, step=1, dtype=None, dev=None):
    dtype = dtype_from_str(dtype)
    return to_dev(_jnp.arange(start, stop, step=step, dtype=dtype), default_device(dev))




def concatenate(xs, axis=-1):
    if xs[0].shape == ():
        return _jnp.concatenate([_jnp.expand_dims(x, 0) for x in xs], axis)
    return _jnp.concatenate(xs, axis)


stack = _jnp.stack







def transpose(x, axes=None):
    if axes is None:
        num_dims = len(x.shape)
        axes = list(range(num_dims))
        axes.reverse()
    return _jnp.transpose(x, axes)


where = lambda condition, x1, x2: _jnp.where(condition, x1, x2)


def indices_where(x):
    where_x = _jnp.where(x)
    ret = _jnp.concatenate([_jnp.expand_dims(item, -1) for item in where_x], -1)
    return ret


reshape = _jnp.reshape
broadcast_to = _jnp.broadcast_to


def squeeze(x, axis=None):
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise Exception('tried to squeeze a zero-dimensional input by axis {}'.format(axis))
    return _jnp.squeeze(x, axis)




# noinspection PyShadowingNames
def zeros_like(x, dtype=None, dev=None):
    if dtype:
        dtype = _jnp.__dict__[dtype]
    else:
        dtype = x.dtype
    return to_dev(_jnp.zeros_like(x, dtype=dtype), default_device(dev))


def full(shape, fill_value, dtype=None, device=None):
    return to_dev(_jnp.full(shape, fill_value, dtype_from_str(default_dtype(dtype, fill_value))),
                  default_device(device))


# noinspection PyUnusedLocal
def one_hot(indices, depth, dev=None):
    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    res = _jnp.eye(depth)[_jnp.array(indices).reshape(-1)]
    return to_dev(res.reshape(list(indices.shape) + [depth]), default_device(dev))


cross = _jnp.cross



# noinspection PyShadowingNames
def identity(n, dtype='float32', batch_shape=None, dev=None):
    dtype = _jnp.__dict__[dtype]
    mat = _jnp.identity(n, dtype=dtype)
    if batch_shape is None:
        return_mat = mat
    else:
        reshape_dims = [1] * len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        return_mat = _jnp.tile(_jnp.reshape(mat, reshape_dims), tile_dims)
    return to_dev(return_mat, default_device(dev))


meshgrid = lambda *xs, indexing='ij': _jnp.meshgrid(*xs, indexing=indexing)



def gather(params, indices, axis=-1, dev=None):
    if dev is None:
        dev = callable_dev(params)
    return to_dev(_jnp.take_along_axis(params, indices, axis), dev)


def gather_nd(params, indices, dev=None):
    if dev is None:
        dev = callable_dev(params)
    indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    res_dim_sizes_list = [_reduce(_mul, params_shape[i + 1:], 1) for i in range(len(params_shape) - 1)] + [1]
    result_dim_sizes = _jnp.array(res_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = _jnp.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = _jnp.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = _jnp.tile(_jnp.reshape(_jnp.sum(indices * indices_scales, -1, keepdims=True), (-1, 1)),
                                       (1, implicit_indices_factor))
    implicit_indices = _jnp.tile(_jnp.expand_dims(_jnp.arange(implicit_indices_factor), 0),
                                 (indices_for_flat_tiled.shape[0], 1))
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = _jnp.reshape(indices_for_flat, (-1,)).astype(_jnp.int32)
    flat_gather = _jnp.take(flat_params, flat_indices_for_flat, 0)
    new_shape = list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    ret = _jnp.reshape(flat_gather, new_shape)
    return to_dev(ret, dev)


def linear_resample(x, num_samples, axis=-1):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    axis = axis % num_x_dims
    x_pre_shape = x_shape[0:axis]
    x_pre_size = _reduce(_mul, x_pre_shape) if x_pre_shape else 1
    num_pre_dims = len(x_pre_shape)
    num_vals = x.shape[axis]
    x_post_shape = x_shape[axis + 1:]
    x_post_size = _reduce(_mul, x_post_shape) if x_post_shape else 1
    num_post_dims = len(x_post_shape)
    xp = _jnp.reshape(_jnp.arange(num_vals * x_pre_size * x_post_size), x_shape)
    x_coords = _jnp.arange(num_samples) * ((num_vals - 1) / (num_samples - 1)) * x_post_size
    x_coords = _jnp.reshape(x_coords, [1] * num_pre_dims + [num_samples] + [1] * num_post_dims)
    x_coords = _jnp.broadcast_to(x_coords, x_pre_shape + [num_samples] + x_post_shape)
    slc = [slice(None)] * num_x_dims
    slc[axis] = slice(0, 1, 1)
    x_coords = x_coords + xp[tuple(slc)]
    x = _jnp.reshape(x, (-1,))
    xp = _jnp.reshape(xp, (-1,))
    x_coords = _jnp.reshape(x_coords, (-1,))
    ret = _jnp.interp(x_coords, xp, x)
    return _jnp.reshape(ret, x_pre_shape + [num_samples] + x_post_shape)


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


compile = lambda fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None: \
    _jax.jit(fn, static_argnums=static_argnums, static_argnames=static_argnames)
current_framework_str = lambda: 'jax'
current_framework_str.__name__ = 'current_framework_str'
multiprocessing = lambda context=None: _multiprocessing if context is None else _multiprocessing.get_context(context)

