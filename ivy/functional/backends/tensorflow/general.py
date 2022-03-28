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
import tensorflow_probability as _tfp
import multiprocessing as _multiprocessing
from tensorflow.python.types.core import Tensor

# local
from ivy.functional.ivy import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.tensorflow import linspace
from ivy.functional.backends.tensorflow.device import _dev_callable, dev_from_str

def is_array(x, exclusive=False):
    if isinstance(x, Tensor):
        if exclusive and isinstance(x, _tf.Variable):
            return False
        return True
    return False


copy_array = _tf.identity
array_equal = _tf.experimental.numpy.array_equal
floormod = lambda x, y: x % y
to_numpy = lambda x: _np.asarray(_tf.convert_to_tensor(x))
to_numpy.__name__ = 'to_numpy'
to_scalar = lambda x: to_numpy(x).item()
to_scalar.__name__ = 'to_scalar'
to_list = lambda x: x.numpy().tolist()
to_list.__name__ = 'to_list'



def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    ret = _tf.unstack(x, axis=axis)
    if keepdims:
        return [_tf.expand_dims(r, axis) for r in ret]
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

cumsum = _tf.cumsum
cumprod = _tf.math.cumprod



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

multiprocessing = lambda context=None: _multiprocessing if context is None else _multiprocessing.get_context(context)
indices_where = _tf.where
shape = lambda x, as_tensor=False: _tf.shape(x) if as_tensor else tuple(x.shape)
shape.__name__ = 'shape'
get_num_dims = lambda x, as_tensor=False: _tf.shape(_tf.shape(x))[0] if as_tensor else int(_tf.shape(_tf.shape(x)))