# global
import numpy as np
import math
from typing import Union, Tuple, Optional, List




def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


def flip(x: np.ndarray,
         axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
         -> np.ndarray:
    num_dims = len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        axis = list(range(num_dims))
    if type(axis) is int:
        axis = [axis]
    axis = [item + num_dims if item < 0 else item for item in axis]
    return np.flip(x, axis)


def expand_dims(x: np.ndarray,
                axis: Optional[Union[int, Tuple[int], List[int]]] = None) \
        -> np.ndarray:
    return np.expand_dims(x, axis)


def permute_dims(x: np.ndarray,
                axes: Tuple[int,...]) \
        -> np.ndarray:
    return np.transpose(x, axes)



# Extra #
# ------#


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
