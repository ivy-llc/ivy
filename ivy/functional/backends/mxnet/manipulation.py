# global
import mxnet as mx
import math
import numpy as np
from typing import Union, Tuple, Optional, List
from ivy.functional.backends.mxnet import _flat_array_to_1_dim_array, _handle_flat_arrays_in_out, _handle_flat_arrays_in, _1_dim_array_to_flat_array


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



def stack(xs, axis=0):
    if xs[0].shape == ():
        return mx.nd.reshape(mx.nd.stack(*[_flat_array_to_1_dim_array(x) for x in xs], axis=axis), -1)
    return mx.nd.stack(*xs, axis=axis)


def squeeze(x, axis=None):
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise Exception('tried to squeeze a zero-dimensional input by axis {}'.format(axis))
    res = mx.nd.squeeze(x, axis)
    if axis is None:
        return _1_dim_array_to_flat_array(res)
    return res


reshape = lambda x, new_shape: x.reshape(new_shape)



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


@_handle_flat_arrays_in
def constant_pad(x, pad_width, value=0):
    if isinstance(pad_width, mx.ndarray.ndarray.NDArray):
        pad_width = pad_width.asnumpy().tolist()
    x_shape = list(x.shape)
    num_dims = len(x_shape)
    if num_dims > 3:
        raise Exception('Invalid inputs. Pad for mxnet only supports inputs with 3 dimensions or smaller.')
    num_dims_to_add = 4 - num_dims
    new_shape = tuple([1] * num_dims_to_add + x_shape)
    mat_expanded_dims = mx.nd.reshape(x, new_shape)
    pad_width_flat = [0]*num_dims_to_add*2 + [item for sublist in pad_width for item in sublist]
    pad_expanded_dims = mx.nd.pad(mat_expanded_dims, mode="constant", pad_width=tuple(pad_width_flat),
                                   constant_value=value)
    new_shape = [orig_dim + pad_width_item[0] + pad_width_item[1] for orig_dim, pad_width_item in zip(x_shape, pad_width)]
    res = mx.nd.reshape(pad_expanded_dims, tuple(new_shape))
    return res


def zero_pad(x, pad_width):
    return constant_pad(x, pad_width, 0)


@_handle_flat_arrays_in_out
def clip(x, x_min, x_max):
    return mx.nd.clip(mx.nd.array(x), x_min, x_max)


swapaxes = mx.nd.swapaxes