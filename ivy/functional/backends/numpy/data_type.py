# global
import numpy as np
from typing import Union, Tuple

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


def result_type(*arrays_and_dtypes: Union[np.ndarray, np.dtype]) -> np.dtype:
    if len(arrays_and_dtypes) <= 1:
        return np.result_type(arrays_and_dtypes)

    result = np.result_type(arrays_and_dtypes[0], arrays_and_dtypes[1])
    for i in range(2, len(arrays_and_dtypes)):
        result = np.result_type(result, arrays_and_dtypes[i])
    return result

  
def broadcast_to(x: np.ndarray, shape: Tuple[int,...]) -> np.ndarray:
    return np.broadcast_to(x,shape)



def cast(x, dtype):
    return x.astype(ivy.dtype_from_str(dtype))


astype = cast
