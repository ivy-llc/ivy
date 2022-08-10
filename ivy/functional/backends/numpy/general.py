"""Collection of Numpy general functions, wrapped to fit Ivy syntax and signature."""

# global
from typing import Optional, Union, Sequence
import numpy as np
from operator import mul
from functools import reduce
import multiprocessing as _multiprocessing
from numbers import Number

# local
import ivy
from ivy.functional.backends.numpy.device import _to_device

# Helpers #
# --------#


def copy_array(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return x.copy()


def array_equal(x0: np.ndarray, x1: np.ndarray) -> bool:
    return np.array_equal(x0, x1)


def to_numpy(x: np.ndarray, copy: bool = True) -> np.ndarray:
    if copy:
        return x.copy()
    else:
        return x


def to_scalar(x: np.ndarray) -> Number:
    return x.item()


def to_list(x: np.ndarray) -> list:
    return x.tolist()


def container_types():
    return []


def inplace_arrays_supported():
    return True


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

    if val_native.shape == x_native.shape:
        np.copyto(x_native, val_native)
    else:
        x_native = val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def is_native_array(x, exclusive=False):
    if isinstance(x, np.ndarray):
        return True
    return False


def floormod(
    x: np.ndarray, y: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
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


def inplace_increment(
    x: Union[ivy.Array, np.ndarray], val: Union[ivy.Array, np.ndarray]
) -> Union[ivy.Array, ivy.Container]:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def cumsum(
    x: np.ndarray, axis: int = 0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.cumsum(x, axis, out=out)


cumsum.support_native_out = True


def cumprod(
    x: np.ndarray,
    axis: int = 0,
    exclusive: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if exclusive:
        x = np.swapaxes(x, axis, -1)
        x = np.concatenate((np.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = np.cumprod(x, -1)
        return np.swapaxes(res, axis, -1)
    return np.cumprod(x, axis, out=out)


cumprod.support_native_out = True


def scatter_flat(
    indices: np.ndarray,
    updates: np.ndarray,
    size: Optional[int] = None,
    tensor: Optional[np.ndarray] = None,
    reduction: str = "sum",
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
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
    return _to_device(target)


# noinspection PyShadowingNames
def scatter_nd(
    indices: np.ndarray,
    updates: np.ndarray,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    tensor: Optional[np.ndarray] = None,
    reduction: str = "sum",
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.to_ivy_shape(target.shape) == ivy.to_ivy_shape(shape)
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
    return _to_device(target)


def gather(
    params: np.ndarray,
    indices: np.ndarray,
    axis: Optional[int] = -1,
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return _to_device(np.take_along_axis(params, indices, axis))


def gather_nd(
    params: np.ndarray, indices: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
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
    return _to_device(res)


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def indices_where(x, out: Optional[np.ndarray] = None):
    where_x = np.where(x)
    if len(where_x) == 1:
        return np.expand_dims(where_x[0], -1)
    res = np.concatenate([np.expand_dims(item, -1) for item in where_x], -1, out=out)
    return res


indices_where.support_native_out = True


# noinspection PyUnusedLocal
def one_hot(
    indices: np.ndarray, depth: int, *, device: str, out: Optional[np.ndarray] = None
) -> np.ndarray:
    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    res = np.eye(depth)[np.array(indices).reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


def shape(x: np.ndarray, as_array: bool = False) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(np.shape(x))
    else:
        return ivy.Shape(x.shape)


def get_num_dims(x, as_tensor=False):
    return np.asarray(len(np.shape(x))) if as_tensor else len(x.shape)


def current_backend_str():
    return "numpy"
