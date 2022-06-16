"""Collection of Numpy general functions, wrapped to fit Ivy syntax and signature."""

# global
from typing import List, Optional, Union
import numpy as np
from operator import mul as mul
from functools import reduce as reduce
import multiprocessing as _multiprocessing
from numbers import Number

# local
import ivy
from ivy.functional.backends.numpy.device import dev, to_device

# Helpers #
# --------#


def copy_array(x: np.ndarray) -> np.ndarray:
    return x.copy()


def array_equal(x0: np.ndarray, x1: np.ndarray) -> bool:
    return np.array_equal(x0, x1)


def to_numpy(x: np.ndarray) -> np.ndarray:
    return x


def to_scalar(x: np.ndarray) -> Number:
    return x.item()


def to_list(x: np.ndarray) -> list:
    return x.tolist()


def container_types():
    return []


inplace_arrays_supported = lambda: True
inplace_variables_supported = lambda: True


def inplace_update(
    x: Union[ivy.Array, np.ndarray],
    val: Union[ivy.Array, np.ndarray],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)

    # make both arrays contiguous if not already
    if not x_native.flags.c_contiguous:
        x_native = np.ascontiguousarray(x_native)
    if not val_native.flags.c_contiguous:
        val_native = np.ascontiguousarray(val_native)

    x_native.data = val_native

    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def is_native_array(x, exclusive=False):
    if isinstance(x, np.ndarray):
        return True
    return False


def floormod(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ret = np.asarray(x % y)
    return ret


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    x_split = np.split(x, x.shape[axis], axis)
    if keepdims:
        return x_split
    return [np.squeeze(item, axis) for item in x_split]


def inplace_decrement(x, val):
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native -= val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_increment(x, val):
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def cumsum(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.cumsum(x, axis)


def cumprod(
    x: np.ndarray, axis: int = 0, exclusive: Optional[bool] = False
) -> np.ndarray:
    if exclusive:
        x = np.swapaxes(x, axis, -1)
        x = np.concatenate((np.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = np.cumprod(x, -1)
        return np.swapaxes(res, axis, -1)
    return np.cumprod(x, axis)


def scatter_flat(indices, updates, size=None, tensor=None, reduction="sum", *, device):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    if device is None:
        device = dev(updates)
    if reduction == "sum":
        if not target_given:
            target = np.zeros([size], dtype=updates.dtype)
        np.add.at(target, indices, updates)
    elif reduction == "replace":
        if not target_given:
            target = np.zeros([size], dtype=updates.dtype)
        target = np.asarray(target).copy()
        target.setflags(write=1)
        target[indices] = updates
    elif reduction == "min":
        if not target_given:
            target = np.ones([size], dtype=updates.dtype) * 1e12
        np.minimum.at(target, indices, updates)
        if not target_given:
            target = np.where(target == 1e12, 0.0, target)
    elif reduction == "max":
        if not target_given:
            target = np.ones([size], dtype=updates.dtype) * -1e12
        np.maximum.at(target, indices, updates)
        if not target_given:
            target = np.where(target == -1e12, 0.0, target)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return to_device(target, device)


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape=None, tensor=None, reduction="sum", *, device):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.shape_to_tuple(target.shape) == ivy.shape_to_tuple(shape)
    if device is None:
        device = dev(updates)
    shape = list(shape) if ivy.exists(shape) else list(tensor.shape)
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    if reduction == "sum":
        if not target_given:
            target = np.zeros(shape, dtype=updates.dtype)
        np.add.at(target, indices_tuple, updates)
    elif reduction == "replace":
        if not target_given:
            target = np.zeros(shape, dtype=updates.dtype)
        target = np.asarray(target).copy()
        target.setflags(write=1)
        target[indices_tuple] = updates
    elif reduction == "min":
        if not target_given:
            target = np.ones(shape, dtype=updates.dtype) * 1e12
        np.minimum.at(target, indices_tuple, updates)
        if not target_given:
            target = np.where(target == 1e12, 0.0, target)
    elif reduction == "max":
        if not target_given:
            target = np.ones(shape, dtype=updates.dtype) * -1e12
        np.maximum.at(target, indices_tuple, updates)
        if not target_given:
            target = np.where(target == -1e12, 0.0, target)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return to_device(target, device)


def gather(
    params: np.ndarray, indices: np.ndarray, axis: Optional[int] = -1, *, device: str
) -> np.ndarray:
    if device is None:
        device = dev(params)
    return to_device(np.take_along_axis(params, indices, axis), device)


def gather_nd(params, indices, *, device: str):
    if device is None:
        device = dev(params)
    indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        reduce(mul, params_shape[i + 1 :], 1) for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = np.array(result_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = np.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = np.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = np.tile(
        np.reshape(np.sum(indices * indices_scales, -1, keepdims=True), (-1, 1)),
        (1, implicit_indices_factor),
    )
    implicit_indices = np.tile(
        np.expand_dims(np.arange(implicit_indices_factor), 0),
        (indices_for_flat_tiled.shape[0], 1),
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = np.reshape(indices_for_flat, (-1,)).astype(np.int32)
    flat_gather = np.take(flat_params, flat_indices_for_flat, 0)
    new_shape = list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    res = np.reshape(flat_gather, new_shape)
    return to_device(res, device)


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def indices_where(x):
    where_x = np.where(x)
    if len(where_x) == 1:
        return np.expand_dims(where_x[0], -1)
    res = np.concatenate([np.expand_dims(item, -1) for item in where_x], -1)
    return res


# noinspection PyUnusedLocal
def one_hot(indices, depth, *, device):
    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    res = np.eye(depth)[np.array(indices).reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


def shape(x: np.ndarray, as_tensor: bool = False) -> Union[np.ndarray, List[int]]:
    if as_tensor:
        return np.asarray(np.shape(x))
    else:
        return x.shape


def get_num_dims(x, as_tensor=False):
    return np.asarray(len(np.shape(x))) if as_tensor else len(x.shape)


def current_backend_str():
    return "numpy"
