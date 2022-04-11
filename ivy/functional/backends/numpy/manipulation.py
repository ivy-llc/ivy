# global
import numpy as np
import math
from typing import Union, Tuple, Optional, List

# local
import ivy


def squeeze(x: np.ndarray,
            axis: Union[int, Tuple[int], List[int]],
            out: Optional[np.ndarray] = None)\
        -> np.ndarray:
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            if ivy.exists(out):
                return ivy.inplace_update(out, x)
            return x
        raise ValueError('tried to squeeze a zero-dimensional input by axis {}'.format(axis))
    ret = np.squeeze(x, axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


def flip(x: np.ndarray,
         axis: Optional[Union[int, Tuple[int], List[int]]] = None,
         out: Optional[np.ndarray] = None)\
         -> np.ndarray:
    num_dims = len(x.shape)
    if not num_dims:
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    if axis is None:
        axis = list(range(num_dims))
    if type(axis) is int:
        axis = [axis]
    axis = [item + num_dims if item < 0 else item for item in axis]
    ret = np.flip(x, axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def expand_dims(x: np.ndarray,
                axis: Optional[Union[int, Tuple[int], List[int]]] = None,
                out: Optional[np.ndarray] = None) \
        -> np.ndarray:
    ret = np.expand_dims(x, axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def permute_dims(x: np.ndarray,
                axes: Tuple[int,...],
                 out: Optional[np.ndarray] = None) \
        -> np.ndarray:
    ret = np.transpose(x, axes)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def concat(xs: List[np.ndarray], axis: int = 0, out: Optional[np.ndarray] = None) -> np.ndarray:
    is_tuple = type(xs) is tuple
    if axis is None:
        if is_tuple:
            xs = list(xs)
        for i in range(len(xs)):
            if xs[i].shape ==():
                xs[i] = np.ravel(xs[i])
        if is_tuple:
            xs = tuple(xs)
    ret = np.concatenate(xs, axis)
    highest_dtype = xs[0].dtype
    for i in xs:
        highest_dtype = np.promote_types(highest_dtype, i.dtype)
    ret = ret.astype(highest_dtype)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def stack(x: Union[Tuple[np.ndarray], List[np.ndarray]],
          axis: Optional[int] = 0)\
          -> np.ndarray:
    return np.stack(x, axis)


def reshape(x: np.ndarray,
            shape: Tuple[int, ...],
            copy: Optional[bool] = None)\
        -> np.ndarray:
    return np.reshape(x, shape)


# Extra #
# ------#
def roll(x: np.ndarray,
         shift: Union[int, Tuple[int, ...]],
         axis: Optional[Union[int, Tuple[int, ...]]] = None) \
        -> np.ndarray:
    return np.roll(x, shift, axis)


def split(x, num_or_size_splits=None, axis=0, with_remainder=False):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception('input array had no shape, but num_sections specified was {}'.format(num_or_size_splits))
        return [x]
    if num_or_size_splits is None:
        num_or_size_splits = x.shape[axis]
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits]*num_chunks_int + [int(remainder*num_or_size_splits)]
    if isinstance(num_or_size_splits, (list, tuple)):
        num_or_size_splits = np.cumsum(num_or_size_splits[:-1])
    return np.split(x, num_or_size_splits, axis)


repeat = np.repeat
tile = np.tile
constant_pad = lambda x, pad_width, value=0: np.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=value)
zero_pad = lambda x, pad_width: np.pad(_flat_array_to_1_dim_array(x), pad_width)
swapaxes = np.swapaxes
clip = lambda x, x_min, x_max: np.asarray(np.clip(x, x_min, x_max))
