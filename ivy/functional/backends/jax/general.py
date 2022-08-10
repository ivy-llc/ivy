"""Collection of Jax general functions, wrapped to fit Ivy syntax and signature."""

# global
import jax
import numpy as np
import jax.numpy as jnp
import jaxlib
from numbers import Number
from operator import mul
from functools import reduce
from jaxlib.xla_extension import Buffer
from typing import Iterable, Optional, Union, Sequence
import multiprocessing as _multiprocessing
from haiku._src.data_structures import FlatMapping

# local
import ivy
from ivy.functional.backends.jax.device import _to_device, _to_array
from ivy.functional.backends.jax import JaxArray


# noinspection PyUnresolvedReferences,PyProtectedMember
def is_native_array(x, exclusive=False):
    if exclusive:
        return isinstance(
            x,
            (
                jax.interpreters.xla._DeviceArray,
                jaxlib.xla_extension.DeviceArray,
                Buffer,
            ),
        )
    return isinstance(
        x,
        (
            jax.interpreters.xla._DeviceArray,
            jaxlib.xla_extension.DeviceArray,
            Buffer,
            jax.interpreters.ad.JVPTracer,
            jax.core.ShapedArray,
            jax.interpreters.partial_eval.DynamicJaxprTracer,
        ),
    )


def copy_array(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.array(x)


def array_equal(x0: JaxArray, x1: JaxArray) -> bool:
    return bool(jnp.array_equal(x0, x1))


def to_numpy(x: JaxArray, copy: bool = True) -> np.ndarray:
    if copy:
        return np.array(_to_array(x))
    else:
        return np.asarray(_to_array(x))


def to_scalar(x: JaxArray) -> Number:
    if isinstance(x, Number):
        return x
    else:
        return _to_array(x).item()


def to_list(x: JaxArray) -> list:
    return _to_array(x).tolist()


def shape(x: JaxArray, as_array: bool = False) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(jnp.shape(x))
    else:
        return ivy.Shape(x.shape)


def get_num_dims(x, as_tensor=False):
    return jnp.asarray(len(jnp.shape(x))) if as_tensor else len(x.shape)


def container_types():
    return [FlatMapping]


def floormod(x: JaxArray, y: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = x % y
    return ret


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    dim_size = x.shape[axis]
    # ToDo: make this faster somehow, jnp.split is VERY slow for large dim_size
    x_split = jnp.split(x, dim_size, axis)
    if keepdims:
        return x_split
    return [jnp.squeeze(item, axis) for item in x_split]


def inplace_update(
    x: Union[ivy.Array, JaxArray],
    val: Union[ivy.Array, JaxArray],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    if ensure_in_backend:
        raise Exception("JAX does not natively support inplace updates")
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_ivy_array(x):
        x.data = val_native
    else:
        raise Exception("JAX does not natively support inplace updates")
    return x


def inplace_arrays_supported():
    return False


inplace_variables_supported = lambda: False


def cumsum(x: JaxArray, axis: int = 0, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.cumsum(x, axis)


def cumprod(
    x: JaxArray,
    axis: int = 0,
    exclusive: Optional[bool] = False,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if exclusive:
        x = jnp.swapaxes(x, axis, -1)
        x = jnp.concatenate((jnp.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = jnp.cumprod(x, -1)
        return jnp.swapaxes(res, axis, -1)
    return jnp.cumprod(x, axis)


def scatter_flat(
    indices: JaxArray,
    updates: JaxArray,
    size: Optional[int] = None,
    tensor: Optional[JaxArray] = None,
    reduction: str = "sum",
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    if reduction == "sum":
        if not target_given:
            target = jnp.zeros([size], dtype=updates.dtype)
        target = target.at[indices].add(updates)
    elif reduction == "replace":
        if not target_given:
            target = jnp.zeros([size], dtype=updates.dtype)
        target = target.at[indices].set(updates)
    elif reduction == "min":
        if not target_given:
            target = jnp.ones([size], dtype=updates.dtype) * 1e12
        target = target.at[indices].min(updates)
        if not target_given:
            target = jnp.where(target == 1e12, 0.0, target)
    elif reduction == "max":
        if not target_given:
            target = jnp.ones([size], dtype=updates.dtype) * -1e12
        target = target.at[indices].max(updates)
        if not target_given:
            target = jnp.where(target == -1e12, 0.0, target)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return _to_device(target)


# noinspection PyShadowingNames
def scatter_nd(
    indices: JaxArray,
    updates: JaxArray,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    tensor: Optional[JaxArray] = None,
    reduction: str = "sum",
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    # parse numeric inputs
    if indices not in [Ellipsis, ()] and not (
        isinstance(indices, Iterable) and Ellipsis in indices
    ):
        indices = [[indices]] if isinstance(indices, Number) else indices
        indices = jnp.array(indices)
        if len(indices.shape) < 2:
            indices = jnp.expand_dims(indices, -1)

    # keep below commented out, array API tests are passing without this
    # updates = [updates] if isinstance(updates, Number) else updates

    updates = jnp.array(
        updates,
        dtype=ivy.dtype(tensor, as_native=True)
        if ivy.exists(tensor)
        else ivy.default_dtype(item=updates),
    )

    # handle Ellipsis
    if isinstance(indices, tuple) or indices is Ellipsis:
        indices_tuple = indices
    else:
        indices_flat = indices.reshape(-1, indices.shape[-1]).T
        indices_tuple = tuple(indices_flat) + (Ellipsis,)

    # implementation
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.to_ivy_shape(target.shape) == ivy.to_ivy_shape(shape)
    shape = list(shape) if ivy.exists(shape) else list(tensor.shape)
    if reduction == "sum":
        if not target_given:
            target = jnp.zeros(shape, dtype=updates.dtype)
        target = target.at[indices_tuple].add(updates)
    elif reduction == "replace":
        if not target_given:
            target = jnp.zeros(shape, dtype=updates.dtype)
        target = target.at[indices_tuple].set(updates)
    elif reduction == "min":
        if not target_given:
            target = jnp.ones(shape, dtype=updates.dtype) * 1e12
        target = target.at[indices_tuple].min(updates)
        if not target_given:
            target = jnp.where(target == 1e12, 0.0, target)
    elif reduction == "max":
        if not target_given:
            target = jnp.ones(shape, dtype=updates.dtype) * -1e12
        target = target.at[indices_tuple].max(updates)
        if not target_given:
            target = jnp.where(target == -1e12, 0.0, target)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return _to_device(target)


def gather(
    params: JaxArray,
    indices: JaxArray,
    axis: int = -1,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return _to_device(jnp.take_along_axis(params, indices, axis))


def gather_nd(
    params: JaxArray, indices: JaxArray, *, out: Optional[JaxArray] = None
) -> JaxArray:
    indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    res_dim_sizes_list = [
        reduce(mul, params_shape[i + 1 :], 1) for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = jnp.array(res_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = jnp.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = jnp.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = jnp.tile(
        jnp.reshape(jnp.sum(indices * indices_scales, -1, keepdims=True), (-1, 1)),
        (1, implicit_indices_factor),
    )
    implicit_indices = jnp.tile(
        jnp.expand_dims(jnp.arange(implicit_indices_factor), 0),
        (indices_for_flat_tiled.shape[0], 1),
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = jnp.reshape(indices_for_flat, (-1,)).astype(jnp.int32)
    flat_gather = jnp.take(flat_params, flat_indices_for_flat, 0)
    new_shape = list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    ret = jnp.reshape(flat_gather, new_shape)
    return _to_device(ret)


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


# noinspection PyUnusedLocal
def one_hot(
    indices: JaxArray, depth: int, *, device, out: Optional[JaxArray] = None
) -> JaxArray:
    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    res = jnp.eye(depth)[jnp.array(indices).reshape(-1)]
    return _to_device(res.reshape(list(indices.shape) + [depth]), device)


def indices_where(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    where_x = jnp.where(x)
    ret = jnp.concatenate([jnp.expand_dims(item, -1) for item in where_x], -1)
    return ret


def inplace_decrement(x, val):
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_ivy_array(x):
        x.data -= val_native
    else:
        x = ivy.Array(val_native)
    return x


def inplace_increment(
    x: Union[ivy.Array, JaxArray], val: Union[ivy.Array, JaxArray]
) -> Union[ivy.Array, ivy.Container]:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_ivy_array(x):
        x.data += val_native
    else:
        x = ivy.Array(val_native)
    return x


current_backend_str = lambda: "jax"
current_backend_str.__name__ = "current_backend_str"
