# global
import math
import tensorflow as tf
from numbers import Number
from typing import Union, Tuple, Optional, List
from tensorflow.python.types.core import Tensor


def flip(x: Tensor,
         axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
         -> Tensor:
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
    return tf.reverse(x, new_axis)


def expand_dims(x: Tensor,
                axis: Optional[Union[int, Tuple[int], List[int]]] = None) \
        -> Tensor:
    try:
        return tf.expand_dims(x, axis)
    except tf.errors.InvalidArgumentError as error:
        raise IndexError(error)


# Extra #
# ------#


def split(x, num_or_size_splits=None, axis=0, with_remainder=False):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception('input array had no shape, but num_sections specified was {}'.format(num_or_size_splits))
        return [x]
    if num_or_size_splits is None:
        dim_size = tf.shape(x)[axis]
        num_or_size_splits = dim_size
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits]*num_chunks_int + [int(remainder*num_or_size_splits)]
    return tf.split(x, num_or_size_splits, axis)


repeat = tf.repeat


def tile(x, reps):
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    if isinstance(reps, Number):
        reps = [reps]
    if isinstance(reps, Tensor) and reps.shape == ():
        reps = tf.reshape(reps, (-1,))
    return tf.tile(x, reps)