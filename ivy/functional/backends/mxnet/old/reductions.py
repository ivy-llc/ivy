"""
Collection of MXNet reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx
from numbers import Number

# local
from ivy.functional.backends.mxnet import _flat_array_to_1_dim_array, _1_dim_array_to_flat_array


def _handle_output(x, axis, keepdims, ret):
    if not keepdims and (axis is None or len((axis,) if isinstance(axis, int) else axis) == len(x.shape)):
        return _1_dim_array_to_flat_array(ret)
    return ret


def einsum(equation, *operands):
    return _mx.np.einsum(equation, *[op.as_np_ndarray() for op in operands]).as_nd_ndarray()
