# global
import numpy as np
from typing import Union

# local
import ivy

import numpy.array_api as npa


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


def can_cast(from_: Union[np.dtype, str, np.ndarray], to: Union[np.dtype, str])\
        -> bool:
    ivy_from_ = ivy.dtype_from_str(from_)
    ivy_to_ = ivy.dtype_from_str(to)

    if isinstance(ivy_from_, np.ndarray):
        ivy_from_ = ivy_from_.dtype

    if ivy_from_ == bool:
        return ivy_from_ == ivy_to_
    elif np.issubdtype(ivy_from_, np.integer) & np.issubdtype(ivy_to_, np.integer):
        from_min, from_max = iinfo(ivy_from_).min, iinfo(ivy_from_).max
        to_min, to_max = iinfo(ivy_to_).min, iinfo(ivy_to_).max
        return from_min >= to_min and from_max <= to_max
    elif np.issubdtype(ivy_from_, np.floating) & np.issubdtype(ivy_to_, np.floating):
        from_min, from_max = finfo(ivy_from_).min, finfo(ivy_from_).max
        to_min, to_max = finfo(ivy_to_).min, finfo(ivy_to_).max
        return from_min >= to_min and from_max <= to_max
    else:
        return False
