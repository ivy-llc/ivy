# global
import math
import tensorflow as tf
from numbers import Number
from typing import Union, Tuple, Optional, List
from tensorflow.python.types.core import Tensor

# local


def roll(
    x: Tensor,
    shift: Union[int, Tuple[int, ...]],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tensor:
    if axis is None:
        originalShape = x.shape
        axis = 0
        x = tf.reshape(x, [-1])
        roll = tf.roll(x, shift, axis)
        ret = tf.reshape(roll, originalShape)
    else:
        if isinstance(shift, int) and (type(axis) in [list, tuple]):
            shift = [shift for _ in range(len(axis))]
        ret = tf.roll(x, shift, axis)
    return ret


def squeeze(x: Tensor, axis: Union[int, Tuple[int], List[int]]) -> Tensor:
    if isinstance(axis, int):
        if x.shape[axis] > 1:
            raise ValueError(
                "Expected dimension of size 1, but found dimension size {}".format(
                    x.shape[axis]
                )
            )
        ret = tf.squeeze(x, axis)
    else:
        if isinstance(axis, tuple):
            axis = list(axis)
        normalise_axis = [
            (len(x.shape) - abs(element)) if element < 0 else element
            for element in axis
        ]
        normalise_axis.sort()
        axis_updated_after_squeeze = [
            dim - key for (key, dim) in enumerate(normalise_axis)
        ]
        for i in axis_updated_after_squeeze:
            if x.shape[i] > 1:
                raise ValueError(
                    "Expected dimension of size 1, but found dimension size {}".format(
                        x.shape[i]
                    )
                )
            else:
                x = tf.squeeze(x, i)
        ret = x
    return ret


def flip(x: Tensor, axis: Optional[Union[int, Tuple[int], List[int]]] = None) -> Tensor:
    num_dims = len(x.shape)
    if not num_dims:
        ret = x
    else:
        if axis is None:
            new_axis = list(range(num_dims))
        else:
            new_axis = axis
        if type(new_axis) is int:
            new_axis = [new_axis]
        else:
            new_axis = new_axis
        new_axis = [item + num_dims if item < 0 else item for item in new_axis]
        ret = tf.reverse(x, new_axis)
    return ret


def expand_dims(x: Tensor, axis: int = 0) -> Tensor:
    try:
        ret = tf.expand_dims(x, axis)
        return ret
    except tf.errors.InvalidArgumentError as error:
        raise IndexError(error)


def permute_dims(x: Tensor, axes: Tuple[int, ...]) -> Tensor:
    ret = tf.transpose(x, perm=axes)
    return ret


def stack(x: Union[Tuple[Tensor], List[Tensor]], axis: Optional[int] = 0) -> Tensor:
    ret = tf.experimental.numpy.stack(x, axis)
    return ret


def reshape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    ret = tf.reshape(x, shape)
    return ret


def concat(xs: List[Tensor], axis: int = 0) -> Tensor:
    is_tuple = type(xs) is tuple
    is_axis_none = axis is None
    if is_tuple:
        xs = list(xs)
    highest_dtype = xs[0].dtype
    for i in xs:
        highest_dtype = tf.experimental.numpy.promote_types(highest_dtype, i.dtype)

    for i in range(len(xs)):
        if is_axis_none:
            xs[i] = tf.reshape(xs[i], -1)
        xs[i] = tf.cast(xs[i], highest_dtype)
    if is_axis_none:
        axis = 0
        if is_tuple:
            xs = tuple(xs)
    ret = tf.concat(xs, axis)
    return ret


# Extra #
# ------#


def split(x, num_or_size_splits=None, axis=0, with_remainder=False):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception(
                "input array had no shape, but num_sections specified was {}".format(
                    num_or_size_splits
                )
            )
        return [x]
    if num_or_size_splits is None:
        dim_size = tf.shape(x)[axis]
        num_or_size_splits = dim_size
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits] * num_chunks_int + [
                int(remainder * num_or_size_splits)
            ]
    return tf.split(x, num_or_size_splits, axis)


def repeat(x: Tensor, repeats: Union[int, List[int]], axis: int = None) -> Tensor:
    ret = tf.repeat(x, repeats, axis)
    return ret


def tile(x, reps):
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    if isinstance(reps, Number):
        reps = [reps]
    if isinstance(reps, Tensor) and reps.shape == ():
        reps = tf.reshape(reps, (-1,))
    ret = tf.tile(x, reps)
    return ret


def constant_pad(x, pad_width, value=0):
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    ret = tf.pad(x, pad_width, constant_values=value)
    return ret


def zero_pad(x, pad_width):
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    ret = tf.pad(x, pad_width)
    return ret


def swapaxes(x, axis0, axis1):
    x_shape = x.shape
    num_dims = len(x_shape)
    axis0 %= num_dims
    axis1 %= num_dims
    config = list(range(num_dims))
    config.pop(axis0)
    config.insert(axis0, axis1)
    config.pop(axis1)
    config.insert(axis1, axis0)
    ret = tf.transpose(x, config)
    return ret


def clip(
    x: Tensor, x_min: Union[Number, Tensor], x_max: Union[Number, Tensor]
) -> Tensor:
    if hasattr(x_min, "dtype") and hasattr(x_max, "dtype"):
        promoted_type = tf.experimental.numpy.promote_types(x.dtype, x_min.dtype)
        promoted_type = tf.experimental.numpy.promote_types(promoted_type, x_max.dtype)
        x = tf.cast(x, promoted_type)
        x_min = tf.cast(x_min, promoted_type)
        x_max = tf.cast(x_max, promoted_type)
    if tf.size(x) == 0:
        ret = x
    elif x.dtype == tf.bool:
        ret = tf.clip_by_value(tf.cast(x, tf.float16), x_min, x_max)
        ret = tf.cast(ret, x.dtype)
    else:
        ret = tf.clip_by_value(x, x_min, x_max)
    return ret
