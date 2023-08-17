# global
import math
from numbers import Number
from typing import Union, Tuple, Optional, List, Sequence, Iterable
import jax.numpy as jnp
import numpy as np

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.jax import JaxArray
from . import backend_version


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


# Array API Standard #
# -------------------#


def concat(
    xs: Union[Tuple[JaxArray, ...], List[JaxArray]],
    /,
    *,
    axis: int = 0,
    out: Optional[JaxArray] = None,
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
    try:
        return jnp.concatenate(xs, axis)
    except ValueError as error:
        raise ivy.utils.exceptions.IvyIndexError(error)


def expand_dims(
    x: JaxArray,
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    try:
        ret = jnp.expand_dims(x, axis)
        return ret
    except ValueError as error:
        raise ivy.utils.exceptions.IvyIndexError(error)


def flip(
    x: JaxArray,
    /,
    *,
    copy: Optional[bool] = None,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.flip(x, axis=axis)


def permute_dims(
    x: JaxArray,
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.transpose(x, axes)


def reshape(
    x: JaxArray,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ivy.utils.assertions.check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = [
            new_s if con else old_s
            for new_s, con, old_s in zip(shape, jnp.array(shape) != 0, x.shape)
        ]
    if copy:
        newarr = jnp.copy(x)
        return jnp.reshape(newarr, shape, order=order)
    return jnp.reshape(x, shape, order=order)


def roll(
    x: JaxArray,
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, jnp.ndarray):
        axis = axis.tolist()
    return jnp.roll(x, shift, axis)


def squeeze(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    copy: Optional[bool] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise ivy.utils.exceptions.IvyException(
            "tried to squeeze a zero-dimensional input by axis {}".format(axis)
        )
    else:
        ret = jnp.squeeze(x, axis=axis)
    return ret


def stack(
    arrays: Union[Tuple[JaxArray], List[JaxArray]],
    /,
    *,
    axis: int = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    try:
        return jnp.stack(arrays, axis=axis)
    except ValueError as error:
        raise ivy.utils.exceptions.IvyIndexError(error)


# Extra #
# ------#


def split(
    x: JaxArray,
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[Union[int, Sequence[int], JaxArray]] = None,
    axis: int = 0,
    with_remainder: bool = False,
) -> List[JaxArray]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise ivy.utils.exceptions.IvyException(
                "input array had no shape, but num_sections specified was {}".format(
                    num_or_size_splits
                )
            )
        return [x]
    if isinstance(num_or_size_splits, jnp.ndarray):
        num_or_size_splits = num_or_size_splits.tolist()
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
        num_or_size_splits = np.cumsum(np.array(num_or_size_splits[:-1]))
    return jnp.split(x, num_or_size_splits, axis)


def repeat(
    x: JaxArray,
    /,
    repeats: Union[int, Iterable[int]],
    *,
    axis: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.repeat(x, repeats, axis)


def tile(
    x: JaxArray, /, repeats: Iterable[int], *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.tile(x, repeats)


def clip(
    x: JaxArray,
    x_min: Union[Number, JaxArray],
    x_max: Union[Number, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
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
            x = x.astype(promoted_type)
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
            x = x.astype(promoted_type)
        else:
            promoted_type = jnp.promote_types(x.dtype, x_min.dtype)
            promoted_type = jnp.promote_types(promoted_type, x_max.dtype)
            x.astype(promoted_type)
    # jnp.clip isn't used because of inconsistent gradients
    if x_min is None and x_max is None:
        raise ValueError("At least one of the x_min or x_max must be provided")
    if x_max is not None:
        x = jnp.where(x > x_max, x_max, x)
    if x_min is not None:
        x = jnp.where(x < x_min, x_min, x)
    return x


@with_unsupported_dtypes({"0.4.14 and below": ("uint64",)}, backend_version)
def constant_pad(
    x: JaxArray,
    /,
    pad_width: List[List[int]],
    *,
    value: Number = 0.0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=value)


def unstack(
    x: JaxArray,
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
) -> List[JaxArray]:
    if x.shape == ():
        return [x]
    dim_size = x.shape[axis]
    # ToDo: make this faster somehow, jnp.split is VERY slow for large dim_size
    x_split = jnp.split(x, dim_size, axis)
    if keepdims:
        return x_split
    return [jnp.squeeze(item, axis) for item in x_split]


def zero_pad(
    x: JaxArray, /, pad_width: List[List[int]], *, out: Optional[JaxArray] = None
):
    return jnp.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=0)


def swapaxes(
    x: JaxArray,
    axis0: int,
    axis1: int,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.swapaxes(x, axis0, axis1)
