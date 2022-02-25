# global
import mxnet as mx
from typing import Union, Tuple

# local
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out

def roll(x: mx.nd.ndarray.NDArray, shift: Union[int, Tuple[int]], axis: Union[int, Tuple[int]]=None) -> mx.nd.ndarray.NDArray:
    return mx.numpy.roll(x, shift, axis)