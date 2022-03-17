# global
import mxnet as mx
from typing import Union, Tuple, Optional, List

from ivy.functional.backends.mxnet import _flat_array_to_1_dim_array


def flip(x: mx.ndarray.ndarray.NDArray,
         axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
         -> mx.ndarray.ndarray.NDArray:
    num_dims = len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        new_axis = list(range(num_dims))
    else:
        new_axis = axis
    if type(new_axis) is int:
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return mx.nd.flip(x, new_axis)


def expand_dims(x: mx.ndarray.ndarray.NDArray,
                axis: Optional[Union[int, Tuple[int], List[int]]] = None) \
        -> mx.ndarray.ndarray.NDArray:
    if x.shape == ():
        return _flat_array_to_1_dim_array(x)
    return mx.nd.expand_dims(x, axis)
