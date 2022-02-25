# global
import numpy as np
from typing import Union, Tuple, Optional

# local
from ivy import dtype_from_str, default_dtype
from ivy.functional.backends.numpy.core.general import _to_dev


# noinspection PyShadowingNames
def zeros(shape: Union[int, Tuple[int, ...]],
          dtype: Optional[np.dtype] = None,
          device: Optional[str] = None) \
        -> np.ndarray:
    return _to_dev(np.zeros(shape, dtype_from_str(default_dtype(dtype))), device)


def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[np.dtype] = None,
         device: Optional[str] = None) \
        -> np.ndarray:
    dtype = dtype_from_str(default_dtype(dtype))
    return _to_dev(np.ones(shape, dtype), device)
