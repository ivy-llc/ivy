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


# noinspection PyShadowingNames
def zeros_like(x: np.ndarray,
               dtype: Optional[np.dtype] =None,
               dev:  Optional[str]  =None)\
            -> np.ndarray:
    if dtype:
        dtype = 'bool_' if dtype == 'bool' else dtype
    else:
        dtype = x.dtype
    return _to_dev(np.zeros_like(x, dtype=dtype), dev)


def ones(shape: Union[int, Tuple[int], List[int]],
         dtype: Optional[np.dtype] = None,
         device: Optional[str] = None) \
        -> np.ndarray:
    dtype = dtype_from_str(default_dtype(dtype))
    return _to_dev(np.ones(shape, dtype), device)
