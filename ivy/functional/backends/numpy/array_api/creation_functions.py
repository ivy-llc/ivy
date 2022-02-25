# global
import numpy as np
from typing import Union, Tuple

# local
from ivy import dtype_from_str, default_dtype
from ivy.functional.backends.numpy.core.general import _to_dev


def zeros(shape, dtype=None) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


# noinspection PyShadowingNames
def zeros(shape: Union[int, Tuple[int, ...]],
          dtype: np.dtype = None,
          device: str = None) \
        -> np.ndarray:
    return _to_dev(np.zeros(shape, dtype_from_str(default_dtype(dtype))), device)
