# global
import ivy
import cupy as cp
import math
from typing import Union, Tuple, Optional, List, Sequence
from numbers import Number


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


# Array API Standard #
# -------------------#


def concat(
    xs: List[cp.ndarray], /, *, axis: int = 0, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    is_tuple = type(xs) is tuple
    if axis is None:
        if is_tuple:
            xs = list(xs)
        for i in range(len(xs)):
            if xs[i].shape == ():
                xs[i] = cp.ravel(xs[i])
        if is_tuple:
            xs = tuple(xs)
    ret = cp.concatenate(xs, axis, out=out)
    highest_dtype = xs[0].dtype
    for i in xs:
        highest_dtype = ivy.as_native_dtype(ivy.promote_types(highest_dtype, i.dtype))
    return ret.astype(highest_dtype)


concat.support_native_out = True


def expand_dims(
    x: cp.ndarray,
    /,
    *,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.expand_dims(x, axis)


def flip(
    x: cp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    num_dims = len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        axis = list(range(num_dims))
    if type(axis) is int:
        axis = [axis]
    axis = [item + num_dims if item < 0 else item for item in axis]
    return cp.flip(x, axis)


def permute_dims(
    x: cp.ndarray, /, axes: Tuple[int, ...], *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.transpose(x, axes)


def reshape(
    x: cp.ndarray,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if copy:
        newarr = x.copy()
        return cp.reshape(newarr, shape)
    return cp.reshape(x, shape)


def roll(
    x: cp.ndarray,
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.roll(x, shift, axis)


def squeeze(
    x: cp.ndarray,
    /,
    axis: Union[int, Sequence[int]],
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise ivy.exceptions.IvyException(
            "tried to squeeze a zero-dimensional input by axis {}".format(axis)
        )
    return cp.squeeze(x, axis=axis)


def stack(
    arrays: Union[Tuple[cp.ndarray], List[cp.ndarray]],
    /,
    *,
    axis: int = 0,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.stack(arrays, axis, out=out)


stack.support_native_out = True


# Extra #
# ------#


def split(
    x: cp.ndarray,
    /,
    *,
    num_or_size_splits: Optional[Union[int, Sequence[int]]] = None,
    axis: Optional[int] = 0,
    with_remainder: Optional[bool] = False,
) -> List[cp.ndarray]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise ivy.exceptions.IvyException(
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
        num_or_size_splits = cp.cumsum(num_or_size_splits[:-1])
    return cp.split(x, num_or_size_splits, axis)


def repeat(
    x: cp.ndarray,
    /,
    repeats: Union[int, List[int]],
    *,
    axis: int = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.repeat(x, repeats, axis)


repeat.unsupported_dtypes = ("uint64",)


def tile(
    x: cp.ndarray, /, reps: Sequence[int], *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.tile(x, reps)


def constant_pad(
    x: cp.ndarray,
    /,
    pad_width: List[List[int]],
    *,
    value: Number = 0.0,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=value)


def zero_pad(
    x: cp.ndarray, /, pad_width: List[List[int]], *, out: Optional[cp.ndarray] = None
):
    return cp.pad(_flat_array_to_1_dim_array(x), pad_width)


def swapaxes(
    x: cp.ndarray, axis0: int, axis1: int, /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.swapaxes(x, axis0, axis1)


def unstack(
    x: cp.ndarray, /, *, axis: int = 0, keepdims: bool = False
) -> List[cp.ndarray]:
    if x.shape == ():
        return [x]
    x_split = cp.split(x, x.shape[axis], axis)
    if keepdims:
        return x_split
    return [cp.squeeze(item, axis) for item in x_split]


def clip(
    x: cp.ndarray,
    x_min: Union[Number, cp.ndarray],
    x_max: Union[Number, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    ivy.assertions.check_less(x_min, x_max, message="min values must be less than max")
    return cp.asarray(cp.clip(x, x_min, x_max, out=out), dtype=x.dtype)


clip.support_native_out = True
