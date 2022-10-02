# global
import cupy as cp
from typing import Union, Optional, Sequence


def all(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.asarray(cp.all(x, axis=axis, keepdims=keepdims, out=out))


all.support_native_out = True


def any(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.asarray(cp.any(x, axis=axis, keepdims=keepdims, out=out))


any.support_native_out = True
