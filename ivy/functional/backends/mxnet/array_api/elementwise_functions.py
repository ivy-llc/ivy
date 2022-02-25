# global
import mxnet as mx
import math

# local
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out


@_handle_flat_arrays_in_out
def isfinite(x: mx.ndarray.ndarray.NDArray)\
        -> mx.ndarray.ndarray.NDArray:
    # ToDo: remove float32 conversion once int8 and uint8 work correctly. Currently 0 returns 0 for these types.
    return mx.nd.contrib.isfinite(x.astype('float32')).astype('bool')

def sqrt(x: mx.ndarray.ndarray.NDArray) -> mx.ndarray.ndarray.NDArray:
    if isinstance(x, float):
        return math.sqrt(x)
    return mx.nd.sqrt(x)
