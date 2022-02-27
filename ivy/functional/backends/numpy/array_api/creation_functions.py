# global
import numpy as np
from typing import Union, Tuple, Optional, List

# local
from ivy import dtype_from_str, default_dtype
# noinspection PyProtectedMember
from ivy.functional.backends.numpy.core.general import _to_dev


def empty(shape: Union[int, Tuple[int], List[int]],
          dtype: Optional[np.dtype] = None,
          device: Optional[str] = None) \
        -> np.ndarray:
    return _to_dev(np.empty(shape, dtype_from_str(default_dtype(dtype))), device)
