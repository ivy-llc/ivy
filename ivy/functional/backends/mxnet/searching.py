# Global
import mxnet as mx
from typing import Optional

# Local
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out


def argmax(
    x: mx.nd.NDArray,
    axis: Optional[int] = None,
    out: Optional[mx.nd.NDArray] = None,
    keepdims: bool = False,
) -> mx.nd.NDArray:
    ret = mx.nd.argmax(x, axis=axis, out=out, keepdims=keepdims)
    return ret


def argmin(
    x: mx.nd.NDArray,
    axis: Optional[int] = None,
    out: Optional[mx.nd.NDArray] = None,
    keepdims: bool = False,
) -> mx.nd.NDArray:
    ret = mx.nd.argmin(x, axis=axis, out=out, keepdims=keepdims)
    return ret


@_handle_flat_arrays_in_out
def where(condition, x1, x2):
    x_shape = list(x1.shape)
    condition_shape = list(condition.shape)
    if x_shape == condition_shape:
        res = mx.nd.where(condition, x1, x2)
        return res
    tile_reps = [int(x / c) for x, c in zip(x_shape, condition_shape)]
    tiled_condition = mx.nd.tile(condition, tile_reps)
    return mx.nd.where(tiled_condition, x1, x2)
