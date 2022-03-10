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


# noinspection SpellCheckingInspection
def full_like(x: np.ndarray,
              fill_value: Union[int, float],
              dtype: Optional[Union[np.dtype, str]] = None,
              device: Optional[str] = None) \
        -> np.ndarray:
    if dtype:
        dtype = 'bool_' if dtype == 'bool' else dtype
    else:
        dtype = x.dtype
    return _to_dev(np.full_like(x, fill_value, dtype=dtype), device)


def tril(x: np.ndarray,
         k: int = 0) \
        -> np.ndarray:
    return np.tril(x, k)
