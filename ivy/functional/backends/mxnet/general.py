# global
import ivy
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
from ivy.functional.ivy import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.mxnet.device import _callable_dev
from ivy.functional.backends.mxnet import linspace
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out, _mxnet_init_context,\
    _scalar_or_flat_array_to_scalar, _handle_flat_arrays_in


def is_array(x, exclusive=False):
    if isinstance(x, _mx.ndarray.ndarray.NDArray):
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
    return _mx.nd.min(_mx.nd.broadcast_equal(x0, x1)) == 1

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
    ret = _mx.nd.split(x, num_outputs, axis, squeeze_axis=not keepdims)
    return ret if isinstance(ret, list) else [ret]

def inplace_update(x, val):
    if x.shape == ():
        raise Exception('MXNet does not support inplace updates of 0-dimensional arrays')
    x[:] = val
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


cumsum = lambda x, axis=0: _mx.nd.cumsum(x, axis if axis >= 0 else axis % len(x.shape))


def cumprod(x, axis=0, exclusive=False):
    array_stack = [_mx.nd.expand_dims(chunk, axis) for chunk in unstack(x, axis)]
    if exclusive:
        array_stack = [_mx.nd.ones_like(array_stack[0])] + array_stack[:-1]
    new_array_list = [array_stack[0]]
    for array_chunk in array_stack[1:]:
        new_array_list.append(new_array_list[-1] * array_chunk)
    return _mx.nd.concat(*new_array_list, dim=axis)
