# global
import numpy as np
from typing import Union

# local
import ivy


# noinspection PyShadowingBuiltins
def iinfo(type: Union[np.dtype, str, np.ndarray])\
        -> np.iinfo:
    return np.iinfo(ivy.dtype_from_str(type))


class Finfo:

    def __init__(self, np_finfo):
        self._np_finfo = np_finfo

    @property
    def bits(self):
        return self._np_finfo.bits

    @property
    def eps(self):
        return float(self._np_finfo.eps)

    @property
    def max(self):
        return float(self._np_finfo.max)

    @property
    def min(self):
        return float(self._np_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._np_finfo.tiny)


# noinspection PyShadowingBuiltins
def finfo(type: Union[np.dtype, str, np.ndarray])\
        -> Finfo:
    return Finfo(np.finfo(ivy.dtype_from_str(type)))
