# global
import ivy
import numpy as np
import math
from typing import Union, Tuple, Optional, List, Sequence
from numbers import Number


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


# Array API Standard #
# -------------------#


def concat(
    xs: List[np.ndarray], axis: int = 0, *, out: Optional[np.ndarray] = None
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
        highest_dtype = np.promote_types(highest_dtype, i.dtype)
    ret = ret.astype(highest_dtype)
    return ret


concat.support_native_out = True


def expand_dims(
    x: np.ndarray,
    axis: Union[int, Tuple[int], List[int]] = 0,
) -> np.ndarray:
    ret = np.expand_dims(x, axis)
    return ret


def flip(
    x: np.ndarray,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    num_dims = len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        axis = list(range(num_dims))
    if type(axis) is int:
        axis = [axis]
    axis = [item + num_dims if item < 0 else item for item in axis]
    ret = np.flip(x, axis)
    return ret


def permute_dims(
    x: np.ndarray, axes: Tuple[int, ...], *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    ret = np.transpose(x, axes)
    return ret


def reshape(
    x: np.ndarray,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
) -> np.ndarray:
    if copy:
        newarr = x.copy()
        return np.reshape(newarr, shape)
    return np.reshape(x, shape)


def roll(
    x: np.ndarray,
    shift: Union[int, Sequence[int]],
    axis: Optional[Union[int, Sequence[int]]] = None,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.roll(x, shift, axis)


def squeeze(
    x: np.ndarray,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise ValueError(
            "tried to squeeze a zero-dimensional input by axis {}".format(axis)
        )
    ret = np.squeeze(x, axis)
    return ret


def stack(
    x: Union[Tuple[np.ndarray], List[np.ndarray]],
    axis: Optional[int] = 0,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.stack(x, axis, out=out)


stack.support_native_out = True


# Extra #
# ------#


def split(
    x,
    num_or_size_splits=None,
    axis=0,
    with_remainder=False,
    *,
    out: Optional[np.ndarray] = None,
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
        num_or_size_splits = np.cumsum(num_or_size_splits[:-1], out=out)
    return np.split(x, num_or_size_splits, axis)


split.support_native_out = True


def repeat(
    x: np.ndarray,
    repeats: Union[int, List[int]],
    axis: int = None,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.repeat(x, repeats, axis)
    return ret


def tile(x: np.ndarray, reps, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    ret = np.tile(x, reps)
    return ret


def constant_pad(
    x: np.ndarray,
    pad_width: List[List[int]],
    value: Number = 0.0,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=value)
    return ret


def zero_pad(
    x: np.ndarray, pad_width: List[List[int]], *, out: Optional[np.ndarray] = None
):
    ret = np.pad(_flat_array_to_1_dim_array(x), pad_width)
    return ret


def swapaxes(
    x: np.ndarray, axis0: int, axis1: int, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    ret = np.swapaxes(x, axis0, axis1)
    return ret


def clip(
    x: np.ndarray,
    x_min: Union[Number, np.ndarray],
    x_max: Union[Number, np.ndarray],
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.asarray(np.clip(x, x_min, x_max))
    return ret
