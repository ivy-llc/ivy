from numbers import Number
import mxnet as mx

#Local
from ivy.functional.backends.mxnet import _flat_array_to_1_dim_array, _1_dim_array_to_flat_array, _handle_output


# Array API Standard #
# -------------------#

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


def std(x, axis=None, keepdims=False):
    red_var = var(x, axis, keepdims)
    is_flat = red_var.shape == ()
    if is_flat:
        red_var = _flat_array_to_1_dim_array(red_var)
    red_std = red_var ** 0.5
    if is_flat:
        return _1_dim_array_to_flat_array(red_std)
    return red_std


def min(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, Number):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        x = _flat_array_to_1_dim_array(x)
    ret = mx.nd.min(x, axis=axis, keepdims=keepdims)
    return _handle_output(x, axis, keepdims, ret)


def max(x, axis=None, keepdims=False):
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


# Extra #
# ------#

def einsum(equation, *operands):
    return mx.np.einsum(equation, *[op.as_np_ndarray() for op in operands]).as_nd_ndarray()
