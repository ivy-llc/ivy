import mxnet as mx
from typing import Optional

# local
import ivy


def argsort(
        x: mx.ndarray.ndarray.NDArray,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.array(mx.nd.argsort(mx.nd.array(x), axis=axis, is_ascend=descending))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sort(
        x: mx.ndarray.ndarray.NDArray,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    kind = "stable" if stable else "quicksort"
    ret = mx.nd.array((mx.nd.sort(mx.nd.sort(x), axis=axis, is_ascend=kind)))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret
