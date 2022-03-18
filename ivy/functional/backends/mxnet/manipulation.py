# global
import mxnet as mx
import math
import numpy as np
from typing import Union, Tuple, Optional, List
from ivy.functional.backends.mxnet import _flat_array_to_1_dim_array, _handle_flat_arrays_in_out

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


# Extra #
# ------#


def split(x, num_or_size_splits=None, axis=0, with_remainder=False):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception('input array had no shape, but num_sections specified was {}'.format(num_or_size_splits))
        return [x]
    if num_or_size_splits == 1:
        return [x]
    elif with_remainder and isinstance(num_or_size_splits, int):
        num_or_size_splits = x.shape[axis] if not num_or_size_splits else num_or_size_splits
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder_size = int((num_chunks - num_chunks_int) * num_or_size_splits)
        num_or_size_splits = [num_or_size_splits]*num_chunks_int + [remainder_size]
    if isinstance(num_or_size_splits, (list, tuple)):
        csum = [0] + np.cumsum(num_or_size_splits).tolist()
        starts = csum[:-1]
        ends = csum[1:]
        if axis < 0:
            slices = [tuple([Ellipsis, slice(s, e, 1)] + [slice(None, None, None)]*int(abs(axis)-1))
                      for s, e in zip(starts, ends)]
        else:
            slices = [tuple([slice(None, None, None)]*axis + [slice(s, e, 1)])
                      for s, e in zip(starts, ends)]
        return [x[so] for so in slices]
    return mx.nd.split(x, x.shape[axis] if not num_or_size_splits else num_or_size_splits, axis)


@_handle_flat_arrays_in_out
def repeat(x, repeats, axis=None):
    return mx.nd.repeat(x, repeats, axis)


def tile(x, reps):
    if isinstance(reps, mx.nd.ndarray.NDArray):
        reps = reps.asnumpy().tolist()
    return mx.nd.tile(_flat_array_to_1_dim_array(x), reps)