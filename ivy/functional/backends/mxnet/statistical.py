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