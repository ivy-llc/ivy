"""
Collection of MXNet reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx
from numbers import Number

# local
from ivy.functional.backends.mxnet.core.general import _flat_array_to_1_dim_array, _1_dim_array_to_flat_array


def _handle_output(x, axis, keepdims, ret):
    if not keepdims and (axis is None or len((axis,) if isinstance(axis, int) else axis) == len(x.shape)):
        return _1_dim_array_to_flat_array(ret)
    return ret


def reduce_sum(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        x = _flat_array_to_1_dim_array(x)
    ret = _mx.nd.sum(x, axis=axis, keepdims=keepdims)
    return _handle_output(x, axis, keepdims, ret)


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        x = _flat_array_to_1_dim_array(x)
    ret = _mx.nd.prod(x, axis=axis, keepdims=keepdims)
    return _handle_output(x, axis, keepdims, ret)


def reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        x = _flat_array_to_1_dim_array(x)
    ret = _mx.nd.mean(x, axis=axis, keepdims=keepdims)
    return _handle_output(x, axis, keepdims, ret)


def reduce_var(x, axis=None, keepdims=False):
    mean_of_x_sqrd = reduce_mean(x ** 2, axis, keepdims)
    mean_of_x = reduce_mean(x, axis, keepdims)
    is_flat = mean_of_x.shape == ()
    if is_flat:
        mean_of_x_sqrd = _flat_array_to_1_dim_array(mean_of_x_sqrd)
        mean_of_x = _flat_array_to_1_dim_array(mean_of_x)
    ret = mean_of_x_sqrd - mean_of_x ** 2
    if is_flat:
        return _1_dim_array_to_flat_array(ret)
    return ret


def reduce_std(x, axis=None, keepdims=False):
    red_var = reduce_var(x, axis, keepdims)
    is_flat = red_var.shape == ()
    if is_flat:
        red_var = _flat_array_to_1_dim_array(red_var)
    red_std = red_var ** 0.5
    if is_flat:
        return _1_dim_array_to_flat_array(red_std)
    return red_std


def reduce_min(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        x = _flat_array_to_1_dim_array(x)
    ret = _mx.nd.min(x, axis=axis, keepdims=keepdims)
    return _handle_output(x, axis, keepdims, ret)


def reduce_max(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        x = _flat_array_to_1_dim_array(x)
    ret = _mx.nd.max(x, axis=axis, keepdims=keepdims)
    return _handle_output(x, axis, keepdims, ret)


def einsum(equation, *operands):
    return _mx.np.einsum(equation, *[op.as_np_ndarray() for op in operands]).as_nd_ndarray()

