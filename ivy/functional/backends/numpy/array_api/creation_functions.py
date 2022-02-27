# global
import numpy as np
from typing import Union, Tuple, Optional, List

# local
from ivy import dtype_from_str, default_dtype
# noinspection PyProtectedMember
from ivy.functional.backends.numpy.core.general import _to_dev


def zeros(shape: Union[int, Tuple[int], List[int]],
          dtype: Optional[np.dtype] = None,
          device: Optional[str] = None) \
        -> np.ndarray:
    return _to_dev(np.zeros(shape, dtype_from_str(default_dtype(dtype))), device)


def ones(shape: Union[int, Tuple[int], List[int]],
         dtype: Optional[np.dtype] = None,
         device: Optional[str] = None) \
        -> np.ndarray:
    dtype = dtype_from_str(default_dtype(dtype))
    return _to_dev(np.ones(shape, dtype), device)


def linspace(start: Union[int, float],
             stop: Union[int, float],
             num: int,
             dtype: Optional[np.dtype] = None,
             device: Optional[str] = None,
             endpoint: bool = True) \
             -> np.ndarray:


    if dtype is None:
        dtype = np.float32
    return _to_dev(np.linspace(np.longdouble(start), np.longdouble(stop), num, dtype=dtype, endpoint=endpoint), device)
