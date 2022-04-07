# global
import ivy
_round = round
import logging
import mxnet as mx
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
from ivy.functional.backends.mxnet import linspace
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out, _mxnet_init_context,\
    _scalar_or_flat_array_to_scalar, _handle_flat_arrays_in


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


def is_native_array(x, exclusive=False):
    if isinstance(x, mx.ndarray.ndarray.NDArray):
        if exclusive and x.grad is not None:
            return False
        return True
    return False


copy_array = lambda x: x.copy()

@_handle_flat_arrays_in_out
def array_equal(x0, x1):
    if ivy.dtype(x0, as_str=True) == 'bool':
        x0 = x0.astype('int32')
    if ivy.dtype(x1, as_str=True) == 'bool':
        x1 = x1.astype('int32')
    return mx.nd.min(mx.nd.broadcast_equal(x0, x1)) == 1

to_numpy = lambda x: x if isinstance(x, _np.ndarray) else (_np.array(x) if isinstance(x, (int, float)) else x.asnumpy())
to_numpy.__name__ = 'to_numpy'
to_scalar = lambda x: x if isinstance(x, Number) else x.asscalar().item()
to_scalar.__name__ = 'to_scalar'
to_list = lambda x: to_numpy(x).tolist()
to_list.__name__ = 'to_list'

@_handle_flat_arrays_in_out
def floormod(x, y):
    return x % y

container_types = lambda: []


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    num_outputs = x.shape[axis]
    ret = mx.nd.split(x, num_outputs, axis, squeeze_axis=not keepdims)
    return ret if isinstance(ret, list) else [ret]


def inplace_update(x, val):
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native[:] = val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


inplace_arrays_supported = lambda: True
inplace_variables_supported = lambda: True


def inplace_decrement(x, val):
    if x.shape == ():
        raise Exception('MXNet does not support inplace updates of 0-dimensional arrays')
    x -= val
    return x


def inplace_increment(x, val):
    if x.shape == ():
        raise Exception('MXNet does not support inplace updates of 0-dimensional arrays')
    x += val
    return x


cumsum = lambda x, axis=0: mx.nd.cumsum(x, axis if axis >= 0 else axis % len(x.shape))


def cumprod(x, axis=0, exclusive=False):
    array_stack = [mx.nd.expand_dims(chunk, axis) for chunk in unstack(x, axis)]
    if exclusive:
        array_stack = [mx.nd.ones_like(array_stack[0])] + array_stack[:-1]
    new_array_list = [array_stack[0]]
    for array_chunk in array_stack[1:]:
        new_array_list.append(new_array_list[-1] * array_chunk)
    return mx.nd.concat(*new_array_list, dim=axis)



# noinspection PyShadowingNames
def scatter_flat(indices, updates, size=None, tensor=None, reduction='sum', dev=None):
    if ivy.exists(tensor):
        raise Exception('MXNet scatter_flat does not support scattering into an pre-existing tensor.')
    if reduction == 'replace':
        return mx.nd.scatter_nd(updates, mx.nd.expand_dims(indices, 0), [size]).copyto(_mxnet_init_context(default_device(dev)))
    else:
        raise Exception('MXNet scatter_flat currently only supports reduction mode "replace", but {} selected.'.
                        format(reduction))


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape=None, tensor=None, reduction='sum', dev=None):
    if ivy.exists(tensor):
        raise Exception('MXNet scatter_flat does not support scattering into an pre-existing tensor.')
    if dev is None:
        dev = _callable_dev(indices)
    shape = list(shape)
    num_idx_dims = len(indices.shape)
    transpose_order = [num_idx_dims-1] + list(range(num_idx_dims-1))
    indices = mx.nd.transpose(indices, transpose_order)
    shape = shape if type(shape) is list else shape.asnumpy().astype(_np.int32).tolist()
    if reduction == 'replace':
        return mx.nd.scatter_nd(updates, indices, shape).copyto(_mxnet_init_context(dev))
    else:
        raise Exception('MXNet scatter_nd currently only supports reduction mode "replace", but {} selected.'.
                        format(reduction))

def gather(params, indices, axis=-1, dev=None):
    if dev is None:
        dev = _callable_dev(params)
    index_slices = unstack(indices, -1)
    res = mx.nd.concat(
        *[mx.nd.expand_dims(mx.nd.pick(params, idx_slice, axis), -1) for idx_slice in index_slices], dim=-1)
    res = mx.nd.reshape(res, indices.shape)
    return res.copyto(_mxnet_init_context(dev))


def gather_nd(params, indices, dev=None):
    if dev is None:
        dev = _callable_dev(params)
    indices_shape = indices.shape
    num_idx_dims = len(indices_shape)
    transpose_order = [num_idx_dims-1] + list(range(num_idx_dims-1))
    indices = mx.nd.transpose(indices, transpose_order)
    return mx.nd.gather_nd(params, indices).copyto(_mxnet_init_context(dev))


multiprocessing = lambda context=None: _multiprocessing if context is None else _multiprocessing.get_context(context)
# noinspection PyUnusedLocal
one_hot = lambda indices, depth, dev=None: mx.nd.one_hot(indices, depth)
shape = lambda x, as_tensor=False: mx.nd.shape_array(x) if as_tensor else x.shape
shape.__name__ = 'shape'
get_num_dims = lambda x, as_tensor=False:\
    mx.nd.shape_array(mx.nd.shape_array(x)).reshape([]) if as_tensor else len(x.shape)

def indices_where(x):
    x_shape = x.shape
    x_flat = x.reshape((1, -1,))
    flat_indices = x_flat.astype('int32').tostype('csr').indices
    if flat_indices.shape == (0,):
        res = flat_indices.reshape((0, len(x_shape)))
        return res
    res = mx.nd.swapaxes(mx.nd.unravel_index(flat_indices, x_shape), 0, 1)
    return res



def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace("<class 'numpy.", '').replace("'>", '').replace('uint', '').replace(
        'int', '').replace('bfloat', '').replace('float', ''))
























# noinspection PyShadowingNames















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

