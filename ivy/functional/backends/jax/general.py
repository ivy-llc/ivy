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
from typing import Iterable, Optional, Union, Sequence, List, Callable
import multiprocessing as _multiprocessing
from haiku._src.data_structures import FlatMapping

# local
import ivy
from ivy.functional.backends.jax.device import _to_device, _to_array
from ivy.functional.backends.jax import JaxArray


def array_equal(x0: JaxArray, x1: JaxArray) -> bool:
    return bool(jnp.array_equal(x0, x1))


def container_types():
    return [FlatMapping]


def copy_array(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.array(x)


def current_backend_str():
    return "jax"


def gather(
    params: JaxArray,
    indices: JaxArray,
    axis: int = -1,
    *,
    out: Optional[JaxArray] = None,
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


def get_num_dims(x: JaxArray, as_tensor: bool = False) -> Union[JaxArray, int]:
    return jnp.asarray(len(jnp.shape(x))) if as_tensor else len(x.shape)


def indices_where(x: JaxArray) -> JaxArray:
    where_x = jnp.where(x)
    return jnp.concatenate([jnp.expand_dims(item, -1) for item in where_x], -1)


def inplace_arrays_supported():
    return False


def inplace_decrement(
    x: Union[ivy.Array, JaxArray], val: Union[ivy.Array, JaxArray]
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_ivy_array(x):
        x.data -= val_native
    else:
        x = ivy.Array(x_native - val_native)
    return x


def inplace_increment(
    x: Union[ivy.Array, JaxArray], val: Union[ivy.Array, JaxArray]
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_ivy_array(x):
        x.data += val_native
    else:
        x = ivy.Array(x_native + val_native)
    return x


def inplace_update(
    x: Union[ivy.Array, JaxArray],
    val: Union[ivy.Array, JaxArray],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    if ivy.is_array(x) and ivy.is_array(val):
        if ensure_in_backend:
            raise Exception("JAX does not natively support inplace updates")
        (x_native, val_native), _ = ivy.args_to_native(x, val)
        if ivy.is_ivy_array(x):
            x.data = val_native
        else:
            raise Exception("JAX does not natively support inplace updates")
        return x
    else:
        return val


def inplace_variables_supported():
    return False


def is_native_array(x, exclusive: bool = False) -> bool:
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


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def one_hot(
    indices: JaxArray,
    depth: int,
    *,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    res = jnp.eye(depth, dtype=indices.dtype)[
        jnp.array(indices, dtype="int64").reshape(-1)
    ]
    return _to_device(res.reshape(list(indices.shape) + [depth]), device)


def scatter_flat(
    indices: JaxArray,
    updates: JaxArray,
    size: Optional[int] = None,
    reduction: str = "sum",
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    target = out
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


def scatter_nd(
    indices: JaxArray,
    updates: JaxArray,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    reduction="sum",
    *,
    out: Optional[JaxArray] = None,
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
        dtype=ivy.dtype(out, as_native=True)
        if ivy.exists(out)
        else ivy.default_dtype(item=updates),
    )

    # handle Ellipsis
    if isinstance(indices, tuple) or indices is Ellipsis:
        indices_tuple = indices
    else:
        indices_flat = indices.reshape(-1, indices.shape[-1]).T
        indices_tuple = tuple(indices_flat) + (Ellipsis,)

    # implementation
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.Shape(target.shape) == ivy.Shape(shape)
    shape = list(shape) if ivy.exists(shape) else list(out.shape)
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
    if ivy.exists(out):
        return ivy.inplace_update(out, _to_device(target))
    return _to_device(target)


scatter_nd.support_native_out = True


def shape(x: JaxArray, as_array: bool = False) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(jnp.shape(x), dtype=ivy.default_int_dtype())
    else:
        return ivy.Shape(x.shape)


def to_list(x: JaxArray) -> list:
    return _to_array(x).tolist()


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


def unstack(x: JaxArray, axis: int, keepdims: bool = False) -> List[JaxArray]:
    if x.shape == ():
        return [x]
    dim_size = x.shape[axis]
    # ToDo: make this faster somehow, jnp.split is VERY slow for large dim_size
    x_split = jnp.split(x, dim_size, axis)
    if keepdims:
        return x_split
    return [jnp.squeeze(item, axis) for item in x_split]


def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: Optional[int] = 0,
) -> Callable:
    return ivy.to_native_arrays_and_back(
        jax.vmap(func, in_axes=in_axes, out_axes=out_axes)
    )
