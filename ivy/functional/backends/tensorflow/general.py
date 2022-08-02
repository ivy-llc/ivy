"""Collection of TensorFlow general functions, wrapped to fit Ivy syntax and
signature.
"""

# global
from typing import Iterable, Optional, Union, Sequence

_round = round
import numpy as np
import multiprocessing as _multiprocessing
from numbers import Number
import tensorflow as tf

# local
import ivy


def is_native_array(x, exclusive=False):
    if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
        if exclusive and isinstance(x, tf.Variable):
            return False
        return True
    return False


def copy_array(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.identity(x)


def array_equal(
    x0: Union[tf.Tensor, tf.Variable],
    x1: Union[tf.Tensor, tf.Variable],
) -> bool:
    return bool((tf.experimental.numpy.array_equal(x0, x1)))


def to_numpy(x: Union[tf.Tensor, tf.Variable]) -> np.ndarray:
    return np.asarray(tf.convert_to_tensor(x))


def to_scalar(x: Union[tf.Tensor, tf.Variable]) -> Number:
    return to_numpy(x).item()


def to_list(x: Union[tf.Tensor, tf.Variable]) -> list:
    return x.numpy().tolist()


def floormod(
    x: Union[tf.Tensor, tf.Variable],
    y: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if hasattr(x, "dtype") and hasattr(y, "dtype"):
        promoted_type = tf.experimental.numpy.promote_types(x.dtype, y.dtype)
        x = tf.cast(x, promoted_type)
        y = tf.cast(y, promoted_type)
    ret = tf.math.floormod(x, y)
    return ret


def unstack(x: Union[tf.Tensor, tf.Variable], axis, keepdims=False):
    if x.shape == ():
        return [x]
    ret = tf.unstack(x, axis=axis)
    if keepdims:
        return [tf.expand_dims(r, axis) for r in ret]
    return ret


def container_types():
    return []


def inplace_update(
    x: Union[ivy.Array, tf.Tensor],
    val: Union[ivy.Array, tf.Tensor],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_variable(x_native):
        x_native.assign(val_native)
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
    elif ensure_in_backend:
        raise Exception("TensorFlow does not support inplace updates of the tf.Tensor")
    elif ivy.is_ivy_array(x):
        x.data = val_native
    else:
        raise Exception("TensorFlow does not support inplace updates of the tf.Tensor")
    return x


def inplace_arrays_supported():
    return False


inplace_variables_supported = lambda: True


def inplace_decrement(x, val):
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_variable(x_native):
        x_native.assign(x_native - val_native)
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
    else:
        if ivy.is_ivy_array(x):
            x.data = val_native
        else:
            x = ivy.Array(val_native)
    return x


def inplace_increment(x, val):
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_variable(x_native):
        x_native.assign(x_native + val_native)
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
    else:
        if ivy.is_ivy_array(x):
            x.data = val_native
        else:
            x = ivy.Array(val_native)
    return x


def cumsum(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = 0,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.cumsum(x, axis)


def cumprod(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = 0,
    exclusive: Optional[bool] = False,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.cumprod(x, axis, exclusive)


# noinspection PyShadowingNames
def scatter_flat(
    indices: Union[tf.Tensor, tf.Variable],
    updates: Union[tf.Tensor, tf.Variable],
    size: Optional[int] = None,
    reduction: str = "sum",
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if indices.dtype != tf.int32 or indices.dtype != tf.int64:
        if indices.dtype in [tf.int8, tf.int16, tf.uint8, tf.uint16]:
            indices = tf.cast(indices, tf.int32)
        else:
            indices = tf.cast(indices, tf.int64)
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    dtype = updates.dtype
    if reduction == "sum":
        if target_given:
            return tf.tensor_scatter_nd_add(
                out, tf.expand_dims(indices, -1), updates
            )
        return tf.scatter_nd(tf.expand_dims(indices, -1), updates, [size])
    elif reduction == "min":
        if not target_given:
            target = tf.fill([size], tf.cast(1e12, dtype))
        res = tf.tensor_scatter_nd_min(target, tf.expand_dims(indices, -1), updates)
        if not target_given:
            res = tf.where(res == tf.cast(1e12, dtype), 0, res)
    elif reduction == "max":
        if not target_given:
            target = tf.fill([size], tf.cast(-1e12, dtype))
        res = tf.tensor_scatter_nd_max(target, tf.expand_dims(indices, -1), updates)
        if not target_given:
            res = tf.where(res == tf.cast(-1e12, dtype), 0, res)
    elif reduction == "replace":
        if target_given:
            res = tf.tensor_scatter_nd_update(
                out, tf.expand_dims(indices, -1), updates
            )
        else:
            res = tf.tensor_scatter_nd_update(
                tf.zeros([size]), tf.expand_dims(indices, -1), updates
            )
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return res


def _parse_ellipsis(so, ndims):
    pre = list()
    for s in so:
        if s is Ellipsis:
            break
        pre.append(s)
    post = list()
    for s in reversed(so):
        if s is Ellipsis:
            break
        post.append(s)
    return tuple(
        pre
        + [slice(None, None, None) for _ in range(ndims - len(pre) - len(post))]
        + list(reversed(post))
    )


# noinspection PyShadowingNames
def scatter_nd(
    indices: Union[tf.Tensor, tf.Variable],
    updates: Union[tf.Tensor, tf.Variable],
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    reduction: str = "sum",
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:

    if ivy.exists(out) and not isinstance(updates, Number):
        out = (
            tf.cast(out, dtype=updates.dtype)
            if ivy.dtype_bits(updates.dtype) > ivy.dtype_bits(out.dtype)
            else out
        )
    # handle numeric updates
    updates = tf.constant(
        # keep below commented out, asarray API tests working without it
        # [updates] if isinstance(updates, Number) else
        updates,
        dtype=ivy.dtype(out, as_native=True)
        if ivy.exists(out)
        else ivy.default_dtype(item=updates),
    )

    # hanle non-tensor indices
    if indices == ():
        return updates
    
    elif indices is Ellipsis or (isinstance(indices, tuple) and indices == (Ellipsis,)):
        if updates.shape == () and ivy.exists(out) and out.shape == ():
            return updates
        shape = out.shape if ivy.exists(out) else updates.shape
        indices =  tf.stack([tf.reshape(value, (-1,)) for value in tf.meshgrid(
                    *[
                        tf.range(shape[0])  
                    ], indexing ='ij'
        )], axis=-1)

    elif isinstance(indices, (tuple, list)) and Ellipsis in indices:
        shape = out.shape if ivy.exists(out) else updates.shape
        indices = _parse_ellipsis(indices, len(shape))
        indices =   tf.stack([tf.reshape(value, (-1,)) for value in tf.meshgrid(
                    *[
                        tf.range(s) if idx == slice(None, None, None) else tf.constant([idx % s])
                        for s, idx in zip(shape, indices)
                    ], indexing ='ij'
        )], axis=-1)        
    else:
        indices = [[indices]] if isinstance(indices, Number) else indices
        indices = tf.constant(indices)
        if len(indices.shape) < 2:
                indices = tf.expand_dims(indices, -1)
        
        if len(updates.shape) < 2:
            updates = tf.expand_dims(updates, 0)
    
    # broadcast updates to indices
    if  updates.shape == ():
        updates = tf.broadcast_to(updates, indices.shape[:1])        
    # implementation
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.Shape(target.shape) == ivy.Shape(shape)
    shape = list(shape) if ivy.exists(shape) else list(out.shape)
    dtype = updates.dtype
    if reduction == "sum":
        if target_given:
            res = tf.tensor_scatter_nd_add(out, indices, updates)
        else:
            res = tf.scatter_nd(indices, updates, shape)
    elif reduction == "min":
        if not target_given:
            target = tf.fill(shape, tf.cast(1e12, dtype))
        res = tf.tensor_scatter_nd_min(target, indices, updates)
        if not target_given:
            res = tf.where(res == tf.cast(1e12, dtype), 0, res)
    elif reduction == "max":
        if not target_given:
            target = tf.fill(shape, tf.cast(-1e12, dtype))
        res = tf.tensor_scatter_nd_max(target, indices, updates)
        if not target_given:
            res = tf.where(res == tf.cast(-1e12, dtype), 0, res)
    elif reduction == "replace":
        if target_given:
            res = tf.tensor_scatter_nd_update(out, indices, updates)
        else:
            res = tf.tensor_scatter_nd_update(tf.zeros(shape, dtype=dtype), indices, updates)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    if ivy.exists(out):
        return ivy.inplace_update(out, res)
    return res

scatter_nd.support_native_out = True

def gather(
    params: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    axis: Optional[int] = -1,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    axis = axis % len(indices.shape)
    return tf.gather(params, indices, axis=axis, batch_dims=axis)


def gather_nd(
    params: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.gather_nd(params, indices)


def one_hot(
    indices: Union[tf.Tensor, tf.Variable],
    depth: int,
    *,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    if indices.dtype == tf.int8:
        indices = tf.cast(indices, tf.uint8)
    elif indices.dtype == tf.int16 or tf.uint16:
        indices = tf.cast(indices, tf.int32)
    else:
        indices = tf.cast(indices, tf.int64)
    device = default_device(device)
    if device is not None:
        with tf.device(as_native_dev(device)):
            return tf.one_hot(indices, depth)
    return tf.one_hot(indices, depth)


def current_backend_str():
    return "tensorflow"


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def indices_where(x: Union[tf.Tensor, tf.Variable]) -> Union[tf.Tensor, tf.Variable]:
    where_x = tf.experimental.numpy.where(x)
    if len(where_x) == 1:
        return tf.expand_dims(where_x[0], -1)
    res = tf.experimental.numpy.concatenate([tf.expand_dims(item, -1) for item in where_x], -1)
    return res

def shape(
    x: Union[tf.Tensor, tf.Variable],
    as_array: bool = False,
) -> Union[tf.Tensor, ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(tf.shape(x))
    else:
        return ivy.Shape(x.shape)


def get_num_dims(x, as_tensor=False):
    return tf.shape(tf.shape(x))[0] if as_tensor else int(tf.shape(tf.shape(x)))
