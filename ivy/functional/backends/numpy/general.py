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
from ivy.functional.backends.numpy.device import _dev_callable, to_dev

# Helpers #
# --------#

def _to_dev(x, dev):
    if dev is not None:
        if 'gpu' in dev:
            raise Exception('Native Numpy does not support GPU placement, consider using Jax instead')
        elif 'cpu' in dev:
            pass
        else:
            raise Exception('Invalid device specified, must be in the form [ "cpu:idx" | "gpu:idx" ],'
                            'but found {}'.format(dev))
    return x


copy_array = lambda x: x.copy()
array_equal = np.array_equal
floormod = lambda x, y: np.asarray(x % y)

to_numpy = lambda x: x
to_numpy.__name__ = 'to_numpy'
to_scalar = lambda x: x.item()
to_scalar.__name__ = 'to_scalar'
to_list = lambda x: x.tolist()
to_list.__name__ = 'to_list'
container_types = lambda: []
inplace_arrays_supported = lambda: True
inplace_variables_supported = lambda: True



def inplace_update(x, val):
    x.data = val
    return x


def is_array(x, exclusive=False):
    if isinstance(x, np.ndarray):
        return True
    return False


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    x_split = np.split(x, x.shape[axis], axis)
    if keepdims:
        return x_split
    return [np.squeeze(item, axis) for item in x_split]


def inplace_decrement(x, val):
    x -= val
    return x


def inplace_increment(x, val):
    x += val
    return x

cumsum = np.cumsum

def cumprod(x, axis=0, exclusive=False):
    if exclusive:
        x = np.swapaxes(x, axis, -1)
        x = np.concatenate((np.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = np.cumprod(x, -1)
        return np.swapaxes(res, axis, -1)
    return np.cumprod(x, axis)



def scatter_flat(indices, updates, size=None, tensor=None, reduction='sum', dev=None):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    if dev is None:
        dev = _dev_callable(updates)
    if reduction == 'sum':
        if not target_given:
            target = np.zeros([size], dtype=updates.dtype)
        np.add.at(target, indices, updates)
    elif reduction == 'replace':
        if not target_given:
            target = np.zeros([size], dtype=updates.dtype)
        target = np.asarray(target).copy()
        target.setflags(write=1)
        target[indices] = updates
    elif reduction == 'min':
        if not target_given:
            target = np.ones([size], dtype=updates.dtype) * 1e12
        np.minimum.at(target, indices, updates)
        if not target_given:
            target = np.where(target == 1e12, 0., target)
    elif reduction == 'max':
        if not target_given:
            target = np.ones([size], dtype=updates.dtype) * -1e12
        np.maximum.at(target, indices, updates)
        if not target_given:
            target = np.where(target == -1e12, 0., target)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    return _to_dev(target, dev)


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape=None, tensor=None, reduction='sum', dev=None):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.shape_to_tuple(target.shape) == ivy.shape_to_tuple(shape)
    if dev is None:
        dev = _dev_callable(updates)
    shape = list(shape) if ivy.exists(shape) else list(tensor.shape)
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    if reduction == 'sum':
        if not target_given:
            target = np.zeros(shape, dtype=updates.dtype)
        np.add.at(target, indices_tuple, updates)
    elif reduction == 'replace':
        if not target_given:
            target = np.zeros(shape, dtype=updates.dtype)
        target = np.asarray(target).copy()
        target.setflags(write=1)
        target[indices_tuple] = updates
    elif reduction == 'min':
        if not target_given:
            target = np.ones(shape, dtype=updates.dtype) * 1e12
        np.minimum.at(target, indices_tuple, updates)
        if not target_given:
            target = np.where(target == 1e12, 0., target)
    elif reduction == 'max':
        if not target_given:
            target = np.ones(shape, dtype=updates.dtype) * -1e12
        np.maximum.at(target, indices_tuple, updates)
        if not target_given:
            target = np.where(target == -1e12, 0., target)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    return _to_dev(target, dev)
