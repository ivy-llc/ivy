"""
Collection of TensorFlow general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
_round = round
import numpy as _np
import math as _math
import tensorflow as _tf
from numbers import Number
from collections import Iterable
import tensorflow_probability as _tfp
import multiprocessing as _multiprocessing
from tensorflow.python.types.core import Tensor

# local
from ivy.functional.ivy.old import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.tensorflow.device import _dev_callable, dev_from_str

DTYPE_TO_STR = {_tf.int8: 'int8',
                _tf.int16: 'int16',
                _tf.int32: 'int32',
                _tf.int64: 'int64',
                _tf.uint8: 'uint8',
                _tf.uint16: 'uint16',
                _tf.uint32: 'uint32',
                _tf.uint64: 'uint64',
                _tf.bfloat16: 'bfloat16',
                _tf.float16: 'float16',
                _tf.float32: 'float32',
                _tf.float64: 'float64',
                _tf.bool: 'bool'}

DTYPE_FROM_STR = {'int8': _tf.int8,
                'int16': _tf.int16,
                'int32': _tf.int32,
                'int64': _tf.int64,
                'uint8': _tf.uint8,
                'uint16': _tf.uint16,
                'uint32': _tf.uint32,
                'uint64': _tf.uint64,
                'bfloat16': _tf.bfloat16,
                'float16': _tf.float16,
                'float32': _tf.float32,
                'float64': _tf.float64,
                'bool': _tf.bool}


# API #
# ----#







def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace('tf.', '').replace('uint', '').replace('int', '').replace('bfloat', '').replace(
        'float', ''))



shape = lambda x, as_tensor=False: _tf.shape(x) if as_tensor else tuple(x.shape)
shape.__name__ = 'shape'
get_num_dims = lambda x, as_tensor=False: _tf.shape(_tf.shape(x))[0] if as_tensor else int(_tf.shape(_tf.shape(x)))
minimum = _tf.minimum
maximum = _tf.maximum
clip = _tf.clip_by_value
# noinspection PyShadowingBuiltins
# noinspection PyShadowingBuiltins
def abs(x):
    if 'uint' in dtype(x, as_str=True):
        return x
    return _tf.abs(x)

def argmin(x, axis=0):
    ret = _tf.argmin(x, axis)
    if ret.shape == ():
        return _tf.reshape(ret, (-1,))
    return ret


def cast(x, dtype):
    return _tf.cast(x, dtype_from_str(dtype))


astype = cast


# noinspection PyShadowingNames
def arange(stop, start=0, step=1, dtype=None, dev=None):
    dtype = _tf.__dict__[dtype] if dtype else dtype
    dev = default_device(dev)
    with _tf.device(dev_from_str(dev)):
        return _tf.range(start, stop, delta=step, dtype=dtype)







def concatenate(xs, axis=-1):
    if xs[0].shape == ():
        return _tf.concat([_tf.expand_dims(x, 0) for x in xs], axis)
    return _tf.concat(xs, axis)


stack = _tf.stack



def constant_pad(x, pad_width, value=0):
    if x.shape == ():
        x = _tf.reshape(x, (-1,))
    return _tf.pad(x, pad_width, constant_values=value)


def zero_pad(x, pad_width):
    if x.shape == ():
        x = _tf.reshape(x, (-1,))
    return _tf.pad(x, pad_width)


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
where = lambda condition, x1, x2: _tf.where(_tf.cast(condition, _tf.bool), x1, x2)
indices_where = _tf.where


reshape = lambda x, newshape: _tf.reshape(x, (newshape,) if isinstance(newshape, int) else newshape)
broadcast_to = _tf.broadcast_to


def squeeze(x, axis=None):
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise Exception('tried to squeeze a zero-dimensional input by axis {}'.format(axis))
    return _tf.squeeze(x, axis)


# noinspection PyShadowingNames
def zeros_like(x, dtype=None, dev=None):
    dtype = _tf.__dict__[dtype] if dtype else dtype
    dev = default_device(dev)
    with _tf.device(dev_from_str(dev)):
        return _tf.zeros_like(x, dtype=dtype)


def full(shape, fill_value, dtype=None, device=None):
    with _tf.device(dev_from_str(default_device(device))):
        return _tf.fill(shape, _tf.constant(fill_value, dtype=dtype_from_str(default_dtype(dtype, fill_value))))


def one_hot(indices, depth, dev=None):
    dev = default_device(dev)
    if dev is not None:
        with _tf.device(dev_from_str(dev)):
            return _tf.one_hot(indices, depth)
    return _tf.one_hot(indices, depth)


cross = _tf.linalg.cross

cumsum = _tf.cumsum
cumprod = _tf.math.cumprod


# noinspection PyShadowingNames
def identity(n, dtype='float32', batch_shape=None, dev=None):
    dtype = _tf.__dict__[dtype]
    dev = default_device(dev)
    with _tf.device(dev_from_str(dev)):
        return _tf.eye(n, n, batch_shape=batch_shape, dtype=dtype)


meshgrid = lambda *xs, indexing='ij': _tf.meshgrid(*xs, indexing=indexing)


# noinspection PyShadowingNames
def scatter_flat(indices, updates, size=None, tensor=None, reduction='sum', dev=None):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    if dev is None:
        dev = _dev_callable(updates)
    dtype = updates.dtype
    if reduction == 'sum':
        if target_given:
            return _tf.tensor_scatter_nd_add(tensor, _tf.expand_dims(indices, -1), updates)
        return _tf.scatter_nd(_tf.expand_dims(indices, -1), updates, [size])
    elif reduction == 'min':
        if not target_given:
            target = _tf.fill([size], _tf.cast(1e12, dtype))
        res = _tf.tensor_scatter_nd_min(target, _tf.expand_dims(indices, -1), updates)
        if not target_given:
            res = _tf.where(res == 1e12, 0., res)
    elif reduction == 'max':
        if not target_given:
            target = _tf.fill([size], _tf.cast(-1e12, dtype))
        res = _tf.tensor_scatter_nd_max(target, _tf.expand_dims(indices, -1), updates)
        if not target_given:
            res = _tf.where(res == -1e12, 0., res)
    elif reduction == 'replace':
        if target_given:
            res = _tf.tensor_scatter_nd_update(tensor, _tf.expand_dims(indices, -1), updates)
        else:
            res = _tf.tensor_scatter_nd_update(_tf.zeros([size]), _tf.expand_dims(indices, -1), updates)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    with _tf.device(dev_from_str(dev)):
        return res


def _parse_ellipsis(so, ndims):
    pre = list()
    for s in so:
        if s is Ellipsis:
            break
        pre.append(s)
    post = list()
    for s in reversed(so):
        if s is Ellipsis:
            break
        post.append(s)
    return tuple(
        pre +
        [slice(None, None, None) for _ in range(ndims - len(pre) - len(post))] +
        list(reversed(post))
    )


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape=None, tensor=None, reduction='sum', dev=None):

    # handle numeric updates
    updates = _tf.constant([updates] if isinstance(updates, Number) else updates,
                           dtype=ivy.dtype(tensor, as_str=False) if ivy.exists(tensor)
                           else ivy.default_dtype(item=updates))

    # hanle non-tensor indices
    if indices == ():
        return updates
    elif indices is Ellipsis or (isinstance(indices, tuple) and indices == (Ellipsis,)):
        if updates.shape == () and ivy.exists(tensor) and tensor.shape == ():
            return updates
        shape = tensor.shape if ivy.exists(tensor) else updates.shape
        indices = _tf.concat([_tf.expand_dims(g, -1) for g in _tf.meshgrid(*[_tf.range(s) for s in shape])], -1)
    elif isinstance(indices, Number):
        indices = (indices,)
    if isinstance(indices, tuple):
        shape = tensor.shape if ivy.exists(tensor) else updates.shape
        indices = _parse_ellipsis(indices, len(shape))
        indices = _tf.concat([_tf.expand_dims(g, -1) for g in _tf.meshgrid(
            *[_tf.range(s) if idx is slice(None, None, None) else idx % s for s, idx in zip(shape, indices)])], -1)

    # broadcast updates to indices
    if updates.shape == ():
        updates = _tf.broadcast_to(updates, indices.shape[:-1])

    # implementation
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.shape_to_tuple(target.shape) == ivy.shape_to_tuple(shape)
    if dev is None:
        dev = _dev_callable(updates)
    shape = list(shape) if ivy.exists(shape) else list(tensor.shape)
    dtype = updates.dtype
    if reduction == 'sum':
        if target_given:
            return _tf.tensor_scatter_nd_add(tensor, indices, updates)
        return _tf.scatter_nd(indices, updates, shape)
    elif reduction == 'min':
        if not target_given:
            target = _tf.fill(shape, _tf.cast(1e12, dtype))
        res = _tf.tensor_scatter_nd_min(target, indices, updates)
        if not target_given:
            res = _tf.where(res == 1e12, 0., res)
    elif reduction == 'max':
        if not target_given:
            target = _tf.fill(shape, _tf.cast(-1e12, dtype))
        res = _tf.tensor_scatter_nd_max(target, indices, updates)
        if not target_given:
            res = _tf.where(res == -1e12, 0., res)
    elif reduction == 'replace':
        if target_given:
            res = _tf.tensor_scatter_nd_update(tensor, indices, updates)
        else:
            res = _tf.tensor_scatter_nd_update(_tf.zeros(shape), indices, updates)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    with _tf.device(dev_from_str(dev)):
        return res


def gather(params, indices, axis=-1, dev=None):
    axis = axis % len(indices.shape)
    if dev is None:
        dev = _dev_callable(params)
    with _tf.device(dev_from_str(dev)):
        return _tf.gather(params, indices, axis=axis, batch_dims=axis)


def gather_nd(params, indices, dev=None):
    if dev is None:
        dev = _dev_callable(params)
    with _tf.device(dev_from_str(dev)):
        return _tf.gather_nd(params, indices)


def linear_resample(x, num_samples, axis=-1):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    axis = axis % num_x_dims
    num_vals = x.shape[axis]
    x_post_shape = x_shape[axis+1:]
    xp = _tf.range(num_vals, dtype=_tf.float32)
    x_coords = _tf.range(num_samples, dtype=_tf.float32) * ((num_vals-1)/(num_samples-1))
    x_coords = x_coords + xp[0:1]
    return _tfp.math.interp_regular_1d_grid(x_coords, 0, num_vals-1, x, axis=axis)


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


compile = lambda fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None: _tf.function(fn)
current_framework_str = lambda: 'tensorflow'
current_framework_str.__name__ = 'current_framework_str'
multiprocessing = lambda context=None: _multiprocessing if context is None else _multiprocessing.get_context(context)






def inplace_decrement(x, val):
    if ivy.is_variable(x):
        x.assign(x - val)
        return x
    raise Exception('TensorFlow does not support inplace operations on non-Variable tensors')


def inplace_increment(x, val):
    if ivy.is_variable(x):
        x.assign(x + val)
        return x
    raise Exception('TensorFlow does not support inplace operations on non-Variable tensors')

