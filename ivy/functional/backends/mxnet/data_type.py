# global
import numpy as np
# noinspection PyPackageRequirements
import mxnet as mx
from typing import Union

# local
import ivy
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out

# noinspection PyShadowingBuiltins
def iinfo(type: Union[type, str, mx.ndarray.ndarray.NDArray])\
        -> np.iinfo:
    return np.iinfo(ivy.dtype_from_str(type))


class Finfo:

    def __init__(self, mx_finfo):
        self._mx_finfo = mx_finfo

    @property
    def bits(self):
        return self._mx_finfo.bits

    @property
    def eps(self):
        return float(self._mx_finfo.eps)

    @property
    def max(self):
        return float(self._mx_finfo.max)

    @property
    def min(self):
        return float(self._mx_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._mx_finfo.tiny)


# noinspection PyShadowingBuiltins
def finfo(type: Union[type, str, mx.ndarray.ndarray.NDArray])\
        -> Finfo:
    return Finfo(np.finfo(ivy.dtype_from_str(type)))

def broadcast_to(x, new_shape):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    num_shape_dims = len(new_shape)
    diff = num_shape_dims - num_x_dims
    if diff == 0:
        return mx.nd.broadcast_to(x, new_shape)
    x = mx.nd.reshape(x, [1]*diff + x_shape)
    return mx.nd.broadcast_to(x, new_shape)


@_handle_flat_arrays_in_out
def astype(x, dtype):
    return x.astype(dtype)
