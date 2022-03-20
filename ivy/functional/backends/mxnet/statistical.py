from numbers import Number
from ivy.functional.backends.mxnet.old.reductions import _handle_output
import mxnet as mx
from ivy.functional.backends.mxnet import _flat_array_to_1_dim_array

def sum(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        x = _flat_array_to_1_dim_array(x)
    ret = mx.nd.sum(x, axis=axis, keepdims=keepdims)
    return _handle_output(x, axis, keepdims, ret)


def prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        x = _flat_array_to_1_dim_array(x)
    ret = mx.nd.prod(x, axis=axis, keepdims=keepdims)
    return _handle_output(x, axis, keepdims, ret)


def mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        x = _flat_array_to_1_dim_array(x)
    ret = mx.nd.mean(x, axis=axis, keepdims=keepdims)
    return _handle_output(x, axis, keepdims, ret)


def var(x, axis=None, keepdims=False):
    mean_of_x_sqrd = mean(x ** 2, axis, keepdims)
    mean_of_x = mean(x, axis, keepdims)
    is_flat = mean_of_x.shape == ()
    if is_flat:
        mean_of_x_sqrd = _flat_array_to_1_dim_array(mean_of_x_sqrd)
        mean_of_x = _flat_array_to_1_dim_array(mean_of_x)
    ret = mean_of_x_sqrd - mean_of_x ** 2
    if is_flat:
        return _1_dim_array_to_flat_array(ret)
    return ret
