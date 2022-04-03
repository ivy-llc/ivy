"""
Collection of TensorFlow general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
_round = round
import numpy as _np
import tensorflow as tf
from numbers import Number
import multiprocessing as _multiprocessing
from tensorflow.python.types.core import Tensor

# local
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.tensorflow.device import _dev_callable, dev_from_str


DTYPE_TO_STR = {tf.int8: 'int8',
                tf.int16: 'int16',
                tf.int32: 'int32',
                tf.int64: 'int64',
                tf.uint8: 'uint8',
                tf.uint16: 'uint16',
                tf.uint32: 'uint32',
                tf.uint64: 'uint64',
                tf.bfloat16: 'bfloat16',
                tf.float16: 'float16',
                tf.float32: 'float32',
                tf.float64: 'float64',
                tf.bool: 'bool'}

DTYPE_FROM_STR = {'int8': tf.int8,
                'int16': tf.int16,
                'int32': tf.int32,
                'int64': tf.int64,
                'uint8': tf.uint8,
                'uint16': tf.uint16,
                'uint32': tf.uint32,
                'uint64': tf.uint64,
                'bfloat16': tf.bfloat16,
                'float16': tf.float16,
                'float32': tf.float32,
                'float64': tf.float64,
                'bool': tf.bool}


def is_native_array(x, exclusive=False):
    if isinstance(x, Tensor):
        if exclusive and isinstance(x, tf.Variable):
            return False
        return True
    return False


copy_array = tf.identity
array_equal = tf.experimental.numpy.array_equal
floormod = lambda x, y: x % y
to_numpy = lambda x: _np.asarray(tf.convert_to_tensor(x))
to_numpy.__name__ = 'to_numpy'
to_scalar = lambda x: to_numpy(x).item()
to_scalar.__name__ = 'to_scalar'
to_list = lambda x: x.numpy().tolist()
to_list.__name__ = 'to_list'


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    ret = tf.unstack(x, axis=axis)
    if keepdims:
        return [tf.expand_dims(r, axis) for r in ret]
    return ret


container_types = lambda: []


def inplace_update(x, val):
    if ivy.is_variable(x):
        x.assign(val)
        return x
    raise Exception('TensorFlow does not support inplace operations on non-Variable tensors')


inplace_arrays_supported = lambda: False
inplace_variables_supported = lambda: True


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


cumsum = tf.cumsum
cumprod = tf.math.cumprod


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
            return tf.tensor_scatter_nd_add(tensor, tf.expand_dims(indices, -1), updates)
        return tf.scatter_nd(tf.expand_dims(indices, -1), updates, [size])
    elif reduction == 'min':
        if not target_given:
            target = tf.fill([size], tf.cast(1e12, dtype))
        res = tf.tensor_scatter_nd_min(target, tf.expand_dims(indices, -1), updates)
        if not target_given:
            res = tf.where(res == 1e12, 0., res)
    elif reduction == 'max':
        if not target_given:
            target = tf.fill([size], tf.cast(-1e12, dtype))
        res = tf.tensor_scatter_nd_max(target, tf.expand_dims(indices, -1), updates)
        if not target_given:
            res = tf.where(res == -1e12, 0., res)
    elif reduction == 'replace':
        if target_given:
            res = tf.tensor_scatter_nd_update(tensor, tf.expand_dims(indices, -1), updates)
        else:
            res = tf.tensor_scatter_nd_update(tf.zeros([size]), tf.expand_dims(indices, -1), updates)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    with tf.device(dev_from_str(dev)):
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
    updates = tf.constant([updates] if isinstance(updates, Number) else updates,
                           dtype=ivy.dtype(tensor, as_str=False) if ivy.exists(tensor)
                           else ivy.default_dtype(item=updates))

    # hanle non-tensor indices
    if indices == ():
        return updates
    elif indices is Ellipsis or (isinstance(indices, tuple) and indices == (Ellipsis,)):
        if updates.shape == () and ivy.exists(tensor) and tensor.shape == ():
            return updates
        shape = tensor.shape if ivy.exists(tensor) else updates.shape
        indices = tf.concat([tf.expand_dims(g, -1) for g in tf.meshgrid(*[tf.range(s) for s in shape])], -1)
    elif isinstance(indices, Number):
        indices = (indices,)
    if isinstance(indices, tuple):
        shape = tensor.shape if ivy.exists(tensor) else updates.shape
        indices = _parse_ellipsis(indices, len(shape))
        indices = tf.concat([tf.expand_dims(g, -1) for g in tf.meshgrid(
            *[tf.range(s) if idx is slice(None, None, None) else idx % s for s, idx in zip(shape, indices)])], -1)

    # broadcast updates to indices
    if updates.shape == ():
        updates = tf.broadcast_to(updates, indices.shape[:-1])

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
            return tf.tensor_scatter_nd_add(tensor, indices, updates)
        return tf.scatter_nd(indices, updates, shape)
    elif reduction == 'min':
        if not target_given:
            target = tf.fill(shape, tf.cast(1e12, dtype))
        res = tf.tensor_scatter_nd_min(target, indices, updates)
        if not target_given:
            res = tf.where(res == 1e12, 0., res)
    elif reduction == 'max':
        if not target_given:
            target = tf.fill(shape, tf.cast(-1e12, dtype))
        res = tf.tensor_scatter_nd_max(target, indices, updates)
        if not target_given:
            res = tf.where(res == -1e12, 0., res)
    elif reduction == 'replace':
        if target_given:
            res = tf.tensor_scatter_nd_update(tensor, indices, updates)
        else:
            res = tf.tensor_scatter_nd_update(tf.zeros(shape), indices, updates)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    with tf.device(dev_from_str(dev)):
        return res


def gather(params, indices, axis=-1, dev=None):
    axis = axis % len(indices.shape)
    if dev is None:
        dev = _dev_callable(params)
    with tf.device(dev_from_str(dev)):
        return tf.gather(params, indices, axis=axis, batch_dims=axis)


def gather_nd(params, indices, dev=None):
    if dev is None:
        dev = _dev_callable(params)
    with tf.device(dev_from_str(dev)):
        return tf.gather_nd(params, indices)


def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace('tf.', '').replace('uint', '').replace('int', '').replace('bfloat', '').replace(
        'float', ''))


def one_hot(indices, depth, dev=None):
    dev = default_device(dev)
    if dev is not None:
        with tf.device(dev_from_str(dev)):
            return tf.one_hot(indices, depth)
    return tf.one_hot(indices, depth)


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


compile = lambda fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None: tf.function(fn)
current_framework_str = lambda: 'tensorflow'
current_framework_str.__name__ = 'current_framework_str'

multiprocessing = lambda context=None: _multiprocessing if context is None else _multiprocessing.get_context(context)
indices_where = tf.where
shape = lambda x, as_tensor=False: tf.shape(x) if as_tensor else tuple(x.shape)
shape.__name__ = 'shape'
get_num_dims = lambda x, as_tensor=False: tf.shape(tf.shape(x))[0] if as_tensor else int(tf.shape(tf.shape(x)))
