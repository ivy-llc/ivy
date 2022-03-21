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
from operator import mul as _mul
from functools import reduce as _reduce
from jaxlib.xla_extension import Buffer
from typing import Iterable
import multiprocessing as _multiprocessing
from haiku._src.data_structures import FlatMapping

# local
import ivy
from ivy.functional.ivy.device import default_device
from ivy.functional.ivy import default_dtype
from ivy.functional.backends.jax.device import to_dev, _to_array, dev as callable_dev

# noinspection PyUnresolvedReferences,PyProtectedMember
def is_array(x, exclusive=False):
    if exclusive:
        return isinstance(x, (_jax.interpreters.xla._DeviceArray,
                              _jaxlib.xla_extension.DeviceArray, Buffer))
    return isinstance(x, (_jax.interpreters.xla._DeviceArray,
                          _jaxlib.xla_extension.DeviceArray, Buffer,
                          _jax.interpreters.ad.JVPTracer,
                          _jax.core.ShapedArray,
                          _jax.interpreters.partial_eval.DynamicJaxprTracer))

copy_array = _jnp.array
array_equal = _jnp.array_equal
floormod = lambda x, y: x % y
to_numpy = lambda x: _onp.asarray(_to_array(x))
to_numpy.__name__ = 'to_numpy'
to_scalar = lambda x: x if isinstance(x, Number) else _to_array(x).item()
to_scalar.__name__ = 'to_scalar'
to_list = lambda x: _to_array(x).tolist()
to_list.__name__ = 'to_list'


container_types = lambda: [FlatMapping]


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    dim_size = x.shape[axis]
    # ToDo: make this faster somehow, jnp.split is VERY slow for large dim_size
    x_split = _jnp.split(x, dim_size, axis)
    if keepdims:
        return x_split
    return [_jnp.squeeze(item, axis) for item in x_split]


def inplace_update(x, val):
    raise Exception('Jax does not support inplace operations')

inplace_arrays_supported = lambda: False
inplace_variables_supported = lambda: False

cumsum = _jnp.cumsum


def cumprod(x, axis=0, exclusive=False):
    if exclusive:
        x = _jnp.swapaxes(x, axis, -1)
        x = _jnp.concatenate((_jnp.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = _jnp.cumprod(x, -1)
        return _jnp.swapaxes(res, axis, -1)
    return _jnp.cumprod(x, axis)


def scatter_flat(indices, updates, size=None, tensor=None, reduction='sum', dev=None):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    if dev is None:
        dev = callable_dev(updates)
    if reduction == 'sum':
        if not target_given:
            target = _jnp.zeros([size], dtype=updates.dtype)
        target = target.at[indices].add(updates)
    elif reduction == 'replace':
        if not target_given:
            target = _jnp.zeros([size], dtype=updates.dtype)
        target = target.at[indices].set(updates)
    elif reduction == 'min':
        if not target_given:
            target = _jnp.ones([size], dtype=updates.dtype) * 1e12
        target = target.at[indices].min(updates)
        if not target_given:
            target = _jnp.where(target == 1e12, 0., target)
    elif reduction == 'max':
        if not target_given:
            target = _jnp.ones([size], dtype=updates.dtype) * -1e12
        target = target.at[indices].max(updates)
        if not target_given:
            target = _jnp.where(target == -1e12, 0., target)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    return to_dev(target, dev)


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape=None, tensor=None, reduction='sum', dev=None):

    # parse numeric inputs
    if indices not in [Ellipsis, ()] and not (isinstance(indices, Iterable) and Ellipsis in indices):
        indices = [[indices]] if isinstance(indices, Number) else indices
        indices = _jnp.array(indices)
        if len(indices.shape) < 2:
            indices = _jnp.expand_dims(indices, -1)
    updates = [updates] if isinstance(updates, Number) else updates
    updates = _jnp.array(updates, dtype=ivy.dtype(tensor, as_str=False) if ivy.exists(tensor)
                         else ivy.default_dtype(item=updates))

    # handle Ellipsis
    if isinstance(indices, tuple) or indices is Ellipsis:
        indices_tuple = indices
    else:
        indices_flat = indices.reshape(-1, indices.shape[-1]).T
        indices_tuple = tuple(indices_flat) + (Ellipsis,)

    # implementation
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.shape_to_tuple(target.shape) == ivy.shape_to_tuple(shape)
    if dev is None:
        dev = callable_dev(updates)
    shape = list(shape) if ivy.exists(shape) else list(tensor.shape)
    if reduction == 'sum':
        if not target_given:
            target = _jnp.zeros(shape, dtype=updates.dtype)
        target = target.at[indices_tuple].add(updates)
    elif reduction == 'replace':
        if not target_given:
            target = _jnp.zeros(shape, dtype=updates.dtype)
        target = target.at[indices_tuple].set(updates)
    elif reduction == 'min':
        if not target_given:
            target = _jnp.ones(shape, dtype=updates.dtype) * 1e12
        target = target.at[indices_tuple].min(updates)
        if not target_given:
            target = _jnp.where(target == 1e12, 0., target)
    elif reduction == 'max':
        if not target_given:
            target = _jnp.ones(shape, dtype=updates.dtype) * -1e12
        target = target.at[indices_tuple].max(updates)
        if not target_given:
            target = _jnp.where(target == -1e12, 0., target)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    return to_dev(target, dev)



def inplace_decrement(x, val):
    raise Exception('Jax does not support inplace operations')


def inplace_increment(x, val):
    raise Exception('Jax does not support inplace operations')
