# global
import numpy as np
from typing import Union, Optional, Sequence, Any


def all(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.asarray(np.all(x, axis=axis, keepdims=keepdims, out=out))


all.support_native_out = True


def any(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.asarray(np.any(x, axis=axis, keepdims=keepdims, out=out))

def is_tensor(x: Any) -> bool:
    return False

any.support_native_out = True
