# global
import numpy as np
# noinspection PyPackageRequirements
import mxnet as mx
from typing import Union

# local
import ivy


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
