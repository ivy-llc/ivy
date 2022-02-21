# global
import numpy as np
import numpy.array_api as npa
from typing import Union, Tuple, Optional, List


# noinspection PyShadowingBuiltins
def all(x: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...], List[int, ...]]] = None,
        keepdims: bool = False)\
        -> np.ndarray:
    return np.asarray(npa.all(npa.asarray(x), axis=axis, keepdims=keepdims))
