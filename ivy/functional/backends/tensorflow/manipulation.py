# global
import math
from numbers import Number
from typing import Union, Tuple, Optional, List, Sequence

import numpy as np
import tensorflow as tf

# local
import ivy

# noinspection PyProtectedMember
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
from ivy.functional.ivy.manipulation import _calculate_out_shape
from . import backend_version


def _reshape_fortran_tf(x, shape):
    if len(x.shape) > 0:
        x = tf.transpose(x)
    return tf.transpose(tf.reshape(x, shape[::-1]))


# Array API Standard #
# -------------------#


def concat(
    xs: Union[Tuple[tf.Tensor, ...], List[tf.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    is_tuple = type(xs) is tuple
    is_axis_none = axis is None
    if is_tuple:
        xs = list(xs)
    highest_dtype = xs[0].dtype
    for i in xs:
        highest_dtype = ivy.as_native_dtype(ivy.promote_types(highest_dtype, i.dtype))

    for i in range(len(xs)):
        if is_axis_none:
            xs[i] = tf.reshape(xs[i], -1)
        xs[i] = ivy.astype(xs[i], highest_dtype, copy=False).to_native()
    if is_axis_none:
        axis = 0
        if is_tuple:
            xs = tuple(xs)
    try:
        return tf.concat(xs, axis)
    except (tf.errors.InvalidArgumentError, np.AxisError) as error:
        raise ivy.utils.exceptions.IvyIndexError(error)


def expand_dims(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    try:
        out_shape = _calculate_out_shape(axis, x.shape)
        ret = tf.reshape(x, shape=out_shape)
        return ret
    except (tf.errors.InvalidArgumentError, np.AxisError) as error:
        raise ivy.utils.exceptions.IvyIndexError(error)


def flip(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
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
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.transpose(x, perm=axes)


def reshape(
    x: Union[tf.Tensor, tf.Variable],
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ivy.utils.assertions.check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = [
            new_s if con else old_s
            for new_s, con, old_s in zip(shape, tf.constant(shape) != 0, x.shape)
        ]
    if copy:
        newarr = tf.experimental.numpy.copy(x)
        if order == "F":
            return _reshape_fortran_tf(newarr, shape)
        return tf.reshape(newarr, shape)
    if order == "F":
        return _reshape_fortran_tf(x, shape)
    return tf.reshape(x, shape)


def roll(
    x: Union[tf.Tensor, tf.Variable],
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
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
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    copy: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if isinstance(axis, int):
        if ivy.any(x.shape[axis] > 1):
            raise ValueError(
                "{} must be lesser than or equal to {}".format(x.shape[axis], 1)
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


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, backend_version)
def stack(
    arrays: Union[Tuple[tf.Tensor], List[tf.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    try:
        return tf.experimental.numpy.stack(arrays, axis)
    except ValueError as e:
        raise ivy.utils.exceptions.IvyIndexError(e)


# Extra #
# ------#


def split(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[
        Union[int, Sequence[int], Union[tf.Tensor, tf.Variable]]
    ] = None,
    axis: int = 0,
    with_remainder: bool = False,
) -> Union[tf.Tensor, tf.Variable]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise ivy.utils.exceptions.IvyException(
                "input array had no shape, but num_sections specified was {}".format(
                    num_or_size_splits
                )
            )
        return [x]
    if num_or_size_splits is None:
        dim_size = tf.shape(x)[axis]
        num_or_size_splits = int(dim_size)
    if isinstance(num_or_size_splits, (tf.Tensor, tf.Variable)):
        num_or_size_splits = tf.cast(num_or_size_splits, tf.int32)
        num_or_size_splits = num_or_size_splits.numpy().tolist()
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits] * num_chunks_int + [
                int(remainder * num_or_size_splits)
            ]

    return tf.split(x, num_or_size_splits, axis)


@with_supported_dtypes({"2.13.0 and below": ("int32", "int64")}, backend_version)
def repeat(
    x: Union[tf.Tensor, tf.Variable],
    /,
    repeats: Union[int, List[int]],
    *,
    axis: int = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.repeat(x, repeats, axis)


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "uint8",
            "uint16",
            "uint32",
            "int8",
            "int16",
        )
    },
    backend_version,
)
def tile(
    x: Union[tf.Tensor, tf.Variable],
    /,
    repeats: Sequence[int],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    if isinstance(repeats, Number):
        repeats = [repeats]
    if isinstance(repeats, tf.Tensor) and repeats.shape == ():
        repeats = tf.reshape(repeats, (-1,))
    # code to unify behaviour with numpy and torch
    if len(x.shape) < len(repeats):
        while len(x.shape) != len(repeats):
            x = tf.expand_dims(x, 0)
    elif len(x.shape) > len(repeats):
        repeats = list(repeats)
        while len(x.shape) != len(repeats):
            repeats = [1] + repeats
    # TODO remove the unifying behaviour code if tensorflow handles this
    # https://github.com/tensorflow/tensorflow/issues/58002
    return tf.tile(x, repeats)


def constant_pad(
    x, /, pad_width, *, value=0, out: Optional[Union[tf.Tensor, tf.Variable]] = None
):
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    return tf.pad(x, pad_width, constant_values=value)


def zero_pad(x, /, pad_width, *, out: Optional[Union[tf.Tensor, tf.Variable]] = None):
    if x.shape == ():
        x = tf.reshape(x, (-1,))
    return tf.pad(x, pad_width)


def swapaxes(
    x,
    axis0,
    axis1,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    x_shape = x.shape
    num_dims = len(x_shape)
    axis0 %= num_dims
    axis1 %= num_dims
    config = list(range(num_dims))
    config.pop(axis0)
    config.insert(axis0, axis1)
    config.pop(axis1)
    config.insert(axis1, axis0)
    return tf.transpose(x, config)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def clip(
    x: Union[tf.Tensor, tf.Variable],
    x_min: Union[Number, tf.Tensor, tf.Variable],
    x_max: Union[Number, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if hasattr(x_min, "dtype") and hasattr(x_max, "dtype"):
        promoted_type = ivy.as_native_dtype(ivy.promote_types(x.dtype, x_min.dtype))
        promoted_type = ivy.as_native_dtype(
            ivy.promote_types(promoted_type, x_max.dtype)
        )
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


def unstack(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
) -> List[tf.Tensor]:
    if x.shape == ():
        return [x]
    ret = tf.unstack(x, axis=axis)
    if keepdims:
        return [tf.expand_dims(r, axis) for r in ret]
    return ret
