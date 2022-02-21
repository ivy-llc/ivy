# global
import numpy as np
from typing import  Union, Tuple, Optional

# local
from ivy.functional.ivy.core import default_dtype
from ivy.functional.backends.numpy import dtype_from_str, _to_dev

# noinspection PyShadowingNames
def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[np.dtype] = 'float32',
         device: Optional[str] = None) \
        -> np.ndarray:
    dtype = dtype_from_str(default_dtype(dtype))
    return _to_dev(np.ones(shape, dtype), device)