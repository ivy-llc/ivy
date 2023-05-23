# global
import math
from numbers import Number
from typing import Union, Tuple, Optional, List, Sequence
import numpy as np

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


# Array API Standard #
# -------------------#


def concat(
    xs: Union[Tuple[np.ndarray, ...], List[np.ndarray]],
    /,
    *,
    axis: int = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    is_tuple = type(xs) is tuple
    if axis is None:
        if is_tuple:
            xs = list(xs)
        for i in range(len(xs)):
            if xs[i].shape == ():
                xs[i] = np.ravel(xs[i])
        if is_tuple:
            xs = tuple(xs)
    ret = np.concatenate(xs, axis, out=out)
    highest_dtype = xs[0].dtype
    for i in xs:
        highest_dtype = ivy.as_native_dtype(ivy.promote_types(highest_dtype, i.dtype))
    return ivy.astype(ret, highest_dtype, copy=False)


concat.support_native_out = True


def expand_dims(
    x: np.ndarray,
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if copy:
        newarr = x.copy()
        return np.expand_dims(newarr, axis)
    return np.expand_dims(x, axis)


def flip(
    x: np.ndarray,
    /,
    *,
    copy: Optional[bool] = None,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    num_dims = len(x.shape)
    if not num_dims:
        if copy:
            newarr = x.copy()
            return newarr
        return x
    if axis is None:
        axis = list(range(num_dims))
    if type(axis) is int:
        axis = [axis]
    axis = [item + num_dims if item < 0 else item for item in axis]
    if copy:
        newarr = x.copy()
        return np.flip(newarr, axis)
    return np.flip(x, axis)


def permute_dims(
    x: np.ndarray,
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if copy:
        newarr = x.copy()
        return np.transpose(newarr, axes)
    return np.transpose(x, axes)


def reshape(
    x: np.ndarray,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ivy.utils.assertions.check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = [
            new_s if con else old_s
            for new_s, con, old_s in zip(shape, np.array(shape) != 0, x.shape)
        ]
    if copy:
        newarr = x.copy()
        return np.reshape(newarr, shape, order=order)
    return np.reshape(x, shape, order=order)


def roll(
    x: np.ndarray,
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.roll(x, shift, axis)


def squeeze(
    x: np.ndarray,
    /,
    axis: Union[int, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise ivy.utils.exceptions.IvyException(
            "tried to squeeze a zero-dimensional input by axis {}".format(axis)
        )
    if copy:
        newarr = x.copy()
        return np.squeeze(newarr, axis=axis)
    return np.squeeze(x, axis=axis)


def stack(
    arrays: Union[Tuple[np.ndarray], List[np.ndarray]],
    /,
    *,
    axis: int = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.stack(arrays, axis, out=out)


stack.support_native_out = True


# Extra #
# ------#


def split(
    x: np.ndarray,
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[Union[int, Sequence[int]]] = None,
    axis: int = 0,
    with_remainder: bool = False,
) -> List[np.ndarray]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise ivy.utils.exceptions.IvyException(
                "input array had no shape, but num_sections specified was {}".format(
                    num_or_size_splits
                )
            )
        if copy:
            newarr = x.copy()
            return [newarr]
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
        num_or_size_splits = np.cumsum(num_or_size_splits[:-1])
    if copy:
        newarr = x.copy()
        return np.split(newarr, num_or_size_splits, axis)
    return np.split(x, num_or_size_splits, axis)


@with_unsupported_dtypes({"1.24.3 and below": ("uint64",)}, backend_version)
def repeat(
    x: np.ndarray,
    /,
    repeats: Union[int, List[int]],
    *,
    axis: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.repeat(x, repeats, axis)


def tile(
    x: np.ndarray, /, repeats: Sequence[int], *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.tile(x, repeats)


def constant_pad(
    x: np.ndarray,
    /,
    pad_width: List[List[int]],
    *,
    value: Number = 0.0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=value)


def zero_pad(
    x: np.ndarray, /, pad_width: List[List[int]], *, out: Optional[np.ndarray] = None
):
    return np.pad(_flat_array_to_1_dim_array(x), pad_width)


def swapaxes(
    x: np.ndarray,
    axis0: int,
    axis1: int,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if copy:
        newarr = x.copy()
        return np.swapaxes(newarr, axis0, axis1)
    return np.swapaxes(x, axis0, axis1)


def unstack(
    x: np.ndarray,
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
) -> List[np.ndarray]:
    if x.shape == ():
        if copy:
            newarr = x.copy()
            return [newarr]
        return [x]
    x_split = None
    if copy:
        newarr = x.copy()
        x_split = np.split(newarr, newarr.shape[axis], axis)
    else:
        x_split = np.split(x, x.shape[axis], axis)
    if keepdims:
        return x_split
    return [np.squeeze(item, axis) for item in x_split]


def clip(
    x: np.ndarray,
    x_min: Union[Number, np.ndarray],
    x_max: Union[Number, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.asarray(np.clip(x, x_min, x_max, out=out), dtype=x.dtype)


clip.support_native_out = True


def as_strided(
    x: np.ndarray,
    shape: Union[ivy.NativeShape, Sequence[int]],
    strides: Sequence[int],
    /,
) -> np.ndarray:
    return np.lib.stride_tricks.as_strided(
        x,
        shape=shape,
        strides=strides,
    )
