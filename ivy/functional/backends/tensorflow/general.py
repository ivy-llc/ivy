"""Collection of TensorFlow general functions, wrapped to fit Ivy syntax and
signature.
"""

# global
from typing import List, Optional, Union

_round = round
import numpy as _np
import tensorflow as tf
import multiprocessing as _multiprocessing
from tensorflow.python.types.core import Tensor
from numbers import Number

# local
import ivy
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.tensorflow.device import _dev_callable, as_native_dev


def is_native_array(x, exclusive=False):
    if isinstance(x, Tensor):
        if exclusive and isinstance(x, tf.Variable):
            return False
        return True
    return False


def copy_array(x: Tensor) -> Tensor:
    return tf.identity(x)


def array_equal(x0: Tensor, x1: Tensor) -> bool:
    return bool((tf.experimental.numpy.array_equal(x0, x1)))


def to_numpy(x: Tensor) -> _np.ndarray:
    return _np.asarray(tf.convert_to_tensor(x))


def to_scalar(x: Tensor) -> Number:
    return to_numpy(x).item()


def to_list(x: Tensor) -> list:
    return x.numpy().tolist()


def floormod(x: tf.Tensor, y: tf.Tensor, out: Optional[tf.Tensor] = None) -> tf.Tensor:
    if hasattr(x, "dtype") and hasattr(y, "dtype"):
        promoted_type = tf.experimental.numpy.promote_types(x.dtype, y.dtype)
        x = tf.cast(x, promoted_type)
        y = tf.cast(y, promoted_type)
    ret = tf.math.floormod(x, y)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    ret = tf.unstack(x, axis=axis)
    if keepdims:
        return [tf.expand_dims(r, axis) for r in ret]
    return ret


container_types = lambda: []


def inplace_update(
    x: Union[ivy.Array, tf.Tensor], val: Union[ivy.Array, tf.Tensor]
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_variable(x_native):
        x_native.assign(val_native)
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


inplace_arrays_supported = lambda: False
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


def cumsum(x: tf.Tensor, axis: int = 0, out: Optional[tf.Tensor] = None) -> tf.Tensor:
    if ivy.exists(out):
        return ivy.inplace_update(out, tf.math.cumsum(x, axis))
    else:
        return tf.math.cumsum(x, axis)


def cumprod(
    x: tf.Tensor,
    axis: int = 0,
    exclusive: Optional[bool] = False,
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    if ivy.exists(out):
        return ivy.inplace_update(out, tf.math.cumprod(x, axis, exclusive))
    else:
        return tf.math.cumprod(x, axis, exclusive)


# noinspection PyShadowingNames
def scatter_flat(indices, updates, size=None, tensor=None, reduction="sum", *, device):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    if device is None:
        device = _dev_callable(updates)
    dtype = updates.dtype
    if reduction == "sum":
        if target_given:
            return tf.tensor_scatter_nd_add(
                tensor, tf.expand_dims(indices, -1), updates
            )
        return tf.scatter_nd(tf.expand_dims(indices, -1), updates, [size])
    elif reduction == "min":
        if not target_given:
            target = tf.fill([size], tf.cast(1e12, dtype))
        res = tf.tensor_scatter_nd_min(target, tf.expand_dims(indices, -1), updates)
        if not target_given:
            res = tf.where(res == 1e12, 0.0, res)
    elif reduction == "max":
        if not target_given:
            target = tf.fill([size], tf.cast(-1e12, dtype))
        res = tf.tensor_scatter_nd_max(target, tf.expand_dims(indices, -1), updates)
        if not target_given:
            res = tf.where(res == -1e12, 0.0, res)
    elif reduction == "replace":
        if target_given:
            res = tf.tensor_scatter_nd_update(
                tensor, tf.expand_dims(indices, -1), updates
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
    with tf.device(as_native_dev(device)):
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
def scatter_nd(indices, updates, shape=None, tensor=None, reduction="sum", *, device):

    if ivy.exists(tensor) and not isinstance(updates, Number):
        tensor = (
            tf.cast(tensor, dtype=updates.dtype)
            if ivy.dtype_bits(updates.dtype) > ivy.dtype_bits(tensor.dtype)
            else tensor
        )
    # handle numeric updates
    updates = tf.constant(
        # keep below commented out, asarray API tests working without it
        # [updates] if isinstance(updates, Number) else
        updates,
        dtype=ivy.dtype(tensor, as_native=True)
        if ivy.exists(tensor)
        else ivy.default_dtype(item=updates),
    )

    # hanle non-tensor indices
    if indices == ():
        return updates
    elif indices is Ellipsis or (isinstance(indices, tuple) and indices == (Ellipsis,)):
        if updates.shape == () and ivy.exists(tensor) and tensor.shape == ():
            return updates
        shape = tensor.shape if ivy.exists(tensor) else updates.shape
        indices = tf.concat(
            [tf.expand_dims(g, -1) for g in tf.meshgrid(*[tf.range(s) for s in shape])],
            -1,
        )
    elif isinstance(indices, Number):
        indices = (indices,)
    if isinstance(indices, tuple):
        shape = tensor.shape if ivy.exists(tensor) else updates.shape
        indices = _parse_ellipsis(indices, len(shape))
        indices = tf.concat(
            [
                tf.expand_dims(g, -1)
                for g in tf.meshgrid(
                    *[
                        tf.range(s) if idx is slice(None, None, None) else idx % s
                        for s, idx in zip(shape, indices)
                    ]
                )
            ],
            -1,
        )

    # broadcast updates to indices
    if updates.shape == ():
        updates = tf.broadcast_to(updates, indices.shape[:-1])

    # implementation
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.shape_to_tuple(target.shape) == ivy.shape_to_tuple(shape)
    if device is None:
        device = _dev_callable(updates)
    shape = list(shape) if ivy.exists(shape) else list(tensor.shape)
    dtype = updates.dtype
    if reduction == "sum":
        if target_given:
            return tf.tensor_scatter_nd_add(tensor, indices, updates)
        return tf.scatter_nd(indices, updates, shape)
    elif reduction == "min":
        if not target_given:
            target = tf.fill(shape, tf.cast(1e12, dtype))
        res = tf.tensor_scatter_nd_min(target, indices, updates)
        if not target_given:
            res = tf.where(res == 1e12, 0.0, res)
    elif reduction == "max":
        if not target_given:
            target = tf.fill(shape, tf.cast(-1e12, dtype))
        res = tf.tensor_scatter_nd_max(target, indices, updates)
        if not target_given:
            res = tf.where(res == -1e12, 0.0, res)
    elif reduction == "replace":
        if target_given:
            res = tf.tensor_scatter_nd_update(tensor, indices, updates)
        else:
            res = tf.tensor_scatter_nd_update(tf.zeros(shape), indices, updates)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    with tf.device(as_native_dev(device)):
        return res


def gather(
    params: tf.Tensor, indices: tf.Tensor, axis: Optional[int] = -1, *, device: str
) -> tf.Tensor:
    axis = axis % len(indices.shape)
    if device is None:
        device = _dev_callable(params)
    with tf.device(as_native_dev(device)):
        return tf.gather(params, indices, axis=axis, batch_dims=axis)


def gather_nd(params, indices, *, device: str):
    if device is None:
        device = _dev_callable(params)
    with tf.device(as_native_dev(device)):
        return tf.gather_nd(params, indices)


def one_hot(indices, depth, *, device):
    device = default_device(device)
    if device is not None:
        with tf.device(as_native_dev(device)):
            return tf.one_hot(indices, depth)
    return tf.one_hot(indices, depth)


current_backend_str = lambda: "tensorflow"
current_backend_str.__name__ = "current_backend_str"

multiprocessing = (
    lambda context=None: _multiprocessing
    if context is None
    else _multiprocessing.get_context(context)
)
indices_where = tf.where


def shape(x: tf.Tensor, as_tensor: bool = False) -> Union[tf.Tensor, List[int]]:
    if as_tensor:
        return tf.shape(x)
    else:
        return tuple(x.shape)


get_num_dims = (
    lambda x, as_tensor=False: tf.shape(tf.shape(x))[0]
    if as_tensor
    else int(tf.shape(tf.shape(x)))
)
