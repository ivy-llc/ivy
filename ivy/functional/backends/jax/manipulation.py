# For Review
# global
import math
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List, Sequence, Iterable
from numbers import Number

# local
import ivy
from ivy.functional.backends.jax import JaxArray


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


# Array API Standard #
# -------------------#


def concat(
    xs: List[JaxArray], /, *, axis: int = 0, out: Optional[JaxArray] = None
) -> JaxArray:
    is_tuple = type(xs) is tuple

    if axis is None:
        if is_tuple:
            xs = list(xs)
        for i in range(len(xs)):
            if xs[i].shape == ():
                xs[i] = jnp.ravel(xs[i])
        if is_tuple:
            xs = tuple(xs)
    return jnp.concatenate(xs, axis)


def expand_dims(
    x: JaxArray,
    /,
    *,
    axis: Union[int, Tuple[int], List[int]] = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    try:
        ret = jnp.expand_dims(x, axis)
        return ret
    except ValueError as error:
        raise IndexError(error)


def flip(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.flip(x, axis=axis)


def permute_dims(
    x: JaxArray, /, axes: Tuple[int, ...], *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.transpose(x, axes)


def reshape(
    x: JaxArray,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if copy:
        newarr = jnp.copy(x)
        return jnp.reshape(newarr, shape)
    return jnp.reshape(x, shape)


def roll(
    x: JaxArray,
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.roll(x, shift, axis)


def squeeze(
    x: JaxArray,
    /,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise ValueError(
            "tried to squeeze a zero-dimensional input by axis {}".format(axis)
        )
    else:
        ret = jnp.squeeze(x, axis=axis)
    return ret


def stack(
    arrays: Union[Tuple[JaxArray], List[JaxArray]],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if axis is None:
        axis = 0
    return jnp.stack(arrays, axis=axis)


# Extra #
# ------#


def split(
    x,
    /,
    *,
    num_or_size_splits=None,
    axis=0,
    with_remainder=False,
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
        num_or_size_splits = x.shape[axis]
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits] * num_chunks_int + [
                int(remainder * num_or_size_splits)
            ]
    if isinstance(num_or_size_splits, (list, tuple)):
        num_or_size_splits = jnp.cumsum(jnp.array(num_or_size_splits[:-1]))
    return jnp.split(x, num_or_size_splits, axis)


def repeat(
    x: JaxArray,
    /,
    repeats: Union[int, Iterable[int]],
    *,
    axis: int = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:

    return jnp.repeat(x, repeats, axis)


def tile(
    x: JaxArray, /, reps: Iterable[int], *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.tile(x, reps)


def clip(
    x: JaxArray,
    x_min: Union[Number, JaxArray],
    x_max: Union[Number, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    assert jnp.all(jnp.less(x_min, x_max)), "Min value must be less than max."
    if (
        hasattr(x_min, "dtype")
        and hasattr(x_max, "dtype")
        and (x.dtype != x_min.dtype or x.dtype != x_max.dtype)
    ):
        if (jnp.float16 in (x.dtype, x_min.dtype, x_max.dtype)) and (
            jnp.int16 in (x.dtype, x_min.dtype, x_max.dtype)
            or jnp.uint16 in (x.dtype, x_min.dtype, x_max.dtype)
        ):
            promoted_type = jnp.promote_types(x.dtype, jnp.float32)
            promoted_type = jnp.promote_types(promoted_type, x_min.dtype)
            promoted_type = jnp.promote_types(promoted_type, x_max.dtype)
            x = jnp.asarray(x, dtype=promoted_type)
        elif (
            jnp.float16 in (x.dtype, x_min.dtype, x_max.dtype)
            or jnp.float32 in (x.dtype, x_min.dtype, x_max.dtype)
        ) and (
            jnp.int32 in (x.dtype, x_min.dtype, x_max.dtype)
            or jnp.uint32 in (x.dtype, x_min.dtype, x_max.dtype)
            or jnp.uint64 in (x.dtype, x_min.dtype, x_max.dtype)
            or jnp.int64 in (x.dtype, x_min.dtype, x_max.dtype)
        ):
            promoted_type = jnp.promote_types(x.dtype, jnp.float64)
            promoted_type = jnp.promote_types(promoted_type, x_min.dtype)
            promoted_type = jnp.promote_types(promoted_type, x_max.dtype)
            x = jnp.asarray(x, dtype=promoted_type)
        else:
            promoted_type = jnp.promote_types(x.dtype, x_min.dtype)
            promoted_type = jnp.promote_types(promoted_type, x_max.dtype)
            x = jnp.asarray(x, dtype=promoted_type)
    return jnp.clip(x, x_min, x_max)


def constant_pad(
    x: JaxArray,
    /,
    pad_width: List[List[int]],
    *,
    value: Number = 0.0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=value)


constant_pad.unsupported_dtypes = ("uint64",)


def zero_pad(
    x: JaxArray, /, pad_width: List[List[int]], *, out: Optional[JaxArray] = None
):
    return jnp.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=0)


def swapaxes(
    x: JaxArray, axis0: int, axis1: int, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.swapaxes(x, axis0, axis1)
