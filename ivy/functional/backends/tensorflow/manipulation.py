# global
import ivy
import math
import tensorflow as tf
from numbers import Number
from typing import Union, Tuple, Optional, List, Sequence

# noinspection PyProtectedMember
from ivy.functional.ivy.manipulation import _calculate_out_shape


# Array API Standard #
# -------------------#


def concat(
    xs: List[tf.Tensor],
    axis: int = 0,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
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


def expand_dims(
    x: Union[tf.Tensor, tf.Variable],
    axis: Union[int, Tuple[int], List[int]] = 0,
) -> Union[tf.Tensor, tf.Variable]:
    try:
        out_shape = _calculate_out_shape(axis, x.shape)
        ret = tf.reshape(x, shape=out_shape)
        return ret
    except tf.errors.InvalidArgumentError as error:
        raise IndexError(error)


def flip(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    *,
    out: Optional[tf.Tensor] = None,
) -> Union[tf.Tensor, tf.Variable]:
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


def permute_dims(
    x: Union[tf.Tensor, tf.Variable],
    axes: Tuple[int, ...],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.transpose(x, perm=axes)
    return ret


def reshape(
    x: Union[tf.Tensor, tf.Variable],
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if copy:
        newarr = tf.experimental.numpy.copy(x)
        return tf.reshape(newarr, shape)
    return tf.reshape(x, shape)


def roll(
    x: Union[tf.Tensor, tf.Variable],
    shift: Union[int, Sequence[int]],
    axis: Optional[Union[int, Sequence[int]]] = None,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
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


def squeeze(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if isinstance(axis, int):
        if x.shape[axis] > 1:
            raise ValueError(
                "Expected dimension of size 1, but found dimension size {}".format(
                    x.shape[axis]
                )
            )
        ret = tf.squeeze(x, axis)
    elif axis is None:
        ret = tf.squeeze(x)
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


def stack(
    x: Union[Tuple[tf.Tensor], List[tf.Tensor]],
    axis: Optional[int] = 0,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.experimental.numpy.stack(x, axis)
    return ret


# Extra #
# ------#


def split(
    x,
    num_or_size_splits=None,
    axis=0,
    with_remainder=False,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
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


def repeat(
    x: Union[tf.Tensor, tf.Variable],
    repeats: Union[int, List[int]],
    axis: int = None,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.repeat(x, repeats, axis)
    return ret


repeat.supported_dtypes = (
    "int32",
    "int64",
)


def tile(x, reps, *, out: Optional[Union[tf.Tensor, tf.Variable]] = None):
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    if isinstance(reps, Number):
        reps = [reps]
    if isinstance(reps, tf.Tensor) and reps.shape == ():
        reps = tf.reshape(reps, (-1,))
    ret = tf.tile(x, reps)
    return ret


tile.unsupported_dtypes = (
    "uint8",
    "uint16",
    "uint32",
    "int8",
    "int16",
)


def constant_pad(
    x, pad_width, value=0, *, out: Optional[Union[tf.Tensor, tf.Variable]] = None
):
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    ret = tf.pad(x, pad_width, constant_values=value)
    return ret


def zero_pad(x, pad_width, *, out: Optional[Union[tf.Tensor, tf.Variable]] = None):
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    ret = tf.pad(x, pad_width)
    return ret


def swapaxes(x, axis0, axis1, *, out: Optional[Union[tf.Tensor, tf.Variable]] = None):
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
    x: Union[tf.Tensor, tf.Variable],
    x_min: Union[Number, tf.Tensor, tf.Variable],
    x_max: Union[Number, tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
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
