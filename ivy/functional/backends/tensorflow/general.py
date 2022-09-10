"""Collection of TensorFlow general functions, wrapped to fit Ivy syntax and
signature.
"""

# global
from typing import Optional, Union, Sequence, List, Callable

_round = round
import numpy as np
import multiprocessing as _multiprocessing
from numbers import Number
import tensorflow as tf

# local
import ivy


def _infer_dtype(x_dtype: tf.DType):
    default_dtype = ivy.infer_default_dtype(x_dtype)
    if ivy.dtype_bits(x_dtype) < ivy.dtype_bits(default_dtype):
        dtype = default_dtype
    else:
        dtype = x_dtype
    return dtype


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


def array_equal(
    x0: Union[tf.Tensor, tf.Variable],
    x1: Union[tf.Tensor, tf.Variable],
) -> bool:
    x0, x1 = ivy.promote_types_of_inputs(x0, x1)
    return bool((tf.experimental.numpy.array_equal(x0, x1)))


def container_types():
    return []


def copy_array(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.identity(x)


def cumprod(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = 0,
    exclusive: Optional[bool] = False,
    *,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    if dtype != x.dtype:
        x = tf.cast(x, dtype)
    return tf.math.cumprod(x, axis, exclusive)


def cumsum(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = 0,
    exclusive: Optional[bool] = False,
    reverse: Optional[bool] = False,
    *,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        if dtype is tf.bool:
            dtype = ivy.default_int_dtype()
        else:
            dtype = _infer_dtype(x.dtype)
    if dtype != x.dtype:
        x = tf.cast(x, dtype)
    return tf.math.cumsum(x, axis, exclusive, reverse)


def current_backend_str():
    return "tensorflow"


def gather(
    params: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    axis: Optional[int] = -1,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = axis % len(indices.shape)
    return tf.gather(params, indices, axis=axis, batch_dims=axis)


def gather_nd(
    params: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.gather_nd(params, indices)


def get_num_dims(x, as_tensor=False):
    return tf.shape(tf.shape(x))[0] if as_tensor else int(tf.shape(tf.shape(x)))


def indices_where(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    where_x = tf.experimental.numpy.where(x)
    if len(where_x) == 1:
        return tf.expand_dims(where_x[0], -1)
    res = tf.experimental.numpy.concatenate(
        [tf.expand_dims(item, -1) for item in where_x], -1
    )
    return res


def inplace_arrays_supported():
    return False


def inplace_decrement(
    x: Union[ivy.Array, tf.Tensor], val: Union[ivy.Array, tf.Tensor]
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    if ivy.is_variable(x_native):
        x_native.assign(x_native - val_native)
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
    else:
        if ivy.is_ivy_array(x):
            x.data -= val_native
        else:
            x = ivy.Array(val_native)
    return x


def inplace_increment(
    x: Union[ivy.Array, tf.Tensor], val: Union[ivy.Array, tf.Tensor]
) -> ivy.Array:
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


def inplace_update(
    x: Union[ivy.Array, tf.Tensor],
    val: Union[ivy.Array, tf.Tensor],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    if ivy.is_array(x) and ivy.is_array(val):
        (x_native, val_native), _ = ivy.args_to_native(x, val)
        if ivy.is_variable(x_native):
            x_native.assign(val_native)
            if ivy.is_ivy_array(x):
                x.data = x_native
            else:
                x = ivy.Array(x_native)
        elif ensure_in_backend:
            raise Exception(
                "TensorFlow does not support inplace updates of the tf.Tensor"
            )
        elif ivy.is_ivy_array(x):
            x.data = val_native
        else:
            raise Exception(
                "TensorFlow does not support inplace updates of the tf.Tensor"
            )
        return x
    else:
        return val


def inplace_variables_supported():
    return True


def is_native_array(x, exclusive=False):
    if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
        if exclusive and isinstance(x, tf.Variable):
            return False
        return True
    return False


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def one_hot(
    indices: Union[tf.Tensor, tf.Variable],
    depth: int,
    *,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    device = ivy.default_device(device)
    dtype = indices.dtype
    if device is not None:
        indices = tf.cast(indices, tf.int64)
        with tf.device(ivy.as_native_dev(device)):
            return tf.one_hot(indices, depth, dtype=dtype)
    return tf.one_hot(indices, depth, dtype=dtype)


def scatter_flat(
    indices: Union[tf.Tensor, tf.Variable],
    updates: Union[tf.Tensor, tf.Variable],
    size: Optional[int] = None,
    reduction: str = "sum",
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
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
            return tf.tensor_scatter_nd_add(out, tf.expand_dims(indices, -1), updates)
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
            res = tf.tensor_scatter_nd_update(out, tf.expand_dims(indices, -1), updates)
        else:
            res = tf.tensor_scatter_nd_update(
                tf.zeros([size], dtype=updates.dtype),
                tf.expand_dims(indices, -1),
                updates,
            )
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return res


def scatter_nd(
    indices: Union[tf.Tensor, tf.Variable],
    updates: Union[tf.Tensor, tf.Variable],
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    reduction: str = "sum",
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
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
        indices = tf.stack(
            [
                tf.reshape(value, (-1,))
                for value in tf.meshgrid(*[tf.range(shape[0])], indexing="ij")
            ],
            axis=-1,
        )

    elif isinstance(indices, (tuple, list)) and Ellipsis in indices:
        shape = out.shape if ivy.exists(out) else updates.shape
        indices = _parse_ellipsis(indices, len(shape))
        indices = tf.stack(
            [
                tf.reshape(value, (-1,))
                for value in tf.meshgrid(
                    *[
                        tf.range(s)
                        if idx == slice(None, None, None)
                        else tf.constant([idx % s])
                        for s, idx in zip(shape, indices)
                    ],
                    indexing="ij",
                )
            ],
            axis=-1,
        )
    else:
        indices = [[indices]] if isinstance(indices, Number) else indices
        indices = tf.constant(indices)
        if len(indices.shape) < 2:
            indices = tf.expand_dims(indices, -1)

        if len(updates.shape) < 2:
            updates = tf.expand_dims(updates, 0)

    # broadcast updates to indices
    if updates.shape == ():
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
            max_value = tf.cast(
                min(
                    tf.experimental.numpy.iinfo(updates.dtype.as_numpy_dtype).max, 1e12
                ),
                updates.dtype,
            )
            target = tf.fill(shape, max_value)
        res = tf.tensor_scatter_nd_min(target, indices, updates)
        if not target_given:
            res = tf.where(res == max_value, 0, res)
    elif reduction == "max":
        if not target_given:
            min_value = tf.cast(
                max(
                    tf.experimental.numpy.iinfo(updates.dtype.as_numpy_dtype).min, -1e12
                ),
                updates.dtype,
            )
            target = tf.fill(shape, min_value)
        res = tf.tensor_scatter_nd_max(target, indices, updates)
        if not target_given:
            res = tf.where(res == min_value, 0, res)
    elif reduction == "replace":
        if target_given:
            res = tf.tensor_scatter_nd_update(out, indices, updates)
        else:
            res = tf.tensor_scatter_nd_update(
                tf.zeros(shape, dtype=dtype), indices, updates
            )
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


def shape(
    x: Union[tf.Tensor, tf.Variable],
    as_array: bool = False,
) -> Union[tf.Tensor, ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(tf.shape(x), dtype=ivy.default_int_dtype())
    else:
        return ivy.Shape(x.shape)


def to_list(x: Union[tf.Tensor, tf.Variable]) -> list:
    return x.numpy().tolist()


def to_numpy(x: Union[tf.Tensor, tf.Variable], copy: bool = True) -> np.ndarray:
    # TensorFlow fails to convert bfloat16 tensor when it has 0 dimensions
    if (
        ivy.is_array(x)
        and get_num_dims(x) == 0
        and ivy.as_native_dtype(x.dtype) is tf.bfloat16
    ):
        x = tf.expand_dims(x, 0)
        if copy:
            return np.squeeze(np.array(tf.convert_to_tensor(x)), 0)
        else:
            return np.squeeze(np.asarray(tf.convert_to_tensor(x)), 0)
    if copy:
        return np.array(tf.convert_to_tensor(x))
    else:
        return np.asarray(tf.convert_to_tensor(x))


def to_scalar(x: Union[tf.Tensor, tf.Variable]) -> Number:
    return to_numpy(x).item()


def unstack(
    x: Union[tf.Tensor, tf.Variable], axis: int, keepdims: bool = False
) -> List[tf.Tensor]:
    if x.shape == ():
        return [x]
    ret = tf.unstack(x, axis=axis)
    if keepdims:
        return [tf.expand_dims(r, axis) for r in ret]
    return ret


def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: Optional[int] = 0,
) -> Callable:
    @ivy.to_native_arrays_and_back
    def _vmap(*args, **kwargs):

        # convert args tuple to list to allow mutability using moveaxis ahead.
        args = list(args)

        # if in_axis is a non-integer, its length should be equal to pos args.
        if isinstance(in_axes, (list, tuple)):
            try:
                assert (len(args)) == len(in_axes)
            except AssertionError:
                raise Exception(
                    """The in_axes should have length equivalent to the 
                number of positional arguments to the function being vectorized
                or it should be an integer."""
                )

        # checking axis_size consistency
        axis_size = set()

        if isinstance(in_axes, int):
            for arg in args:
                axis_size.add(arg.shape[in_axes])
        elif isinstance(in_axes, (list, tuple)):
            for arg, axis in zip(args, in_axes):
                if axis is not None:
                    axis_size.add(arg.shape[axis])

        if len(axis_size) > 1:
            raise ValueError(
                """Inconsistent sizes. All mapped axes should have the same size"""
            )

        # Making sure not all in_axes are None
        if isinstance(in_axes, (list, tuple)):
            assert not all(
                ax is None for ax in in_axes
            ), "At least one of the axes should be specified (not None)."
        else:
            assert not (in_axes is None), "single value in_axes should not be None."

        # Handling None in in_axes by broadcasting the axis_size
        if isinstance(in_axes, (tuple, list)) and None in in_axes:
            none_axis_index = list()
            for index, axis in enumerate(in_axes):
                if axis is None:
                    none_axis_index.append(index)

            for none_mapped_axis in none_axis_index:
                args[none_mapped_axis] = tf.broadcast_to(
                    args[none_mapped_axis],
                    (tuple(axis_size) + args[none_mapped_axis].shape),
                )

        # set up the axis to be mapped
        if isinstance(in_axes, (tuple, list)):
            for i in range(len(in_axes)):
                args[i] = tf.experimental.numpy.moveaxis(args[i], in_axes[i], 0)
        elif isinstance(in_axes, int):
            args[0] = tf.experimental.numpy.moveaxis(args[0], in_axes, 0)

        # vectorisation - applying map_fn if only one arg provided as reduce requires
        # two elements to begin with.
        arr_results = []
        for arrays in zip(*args):
            single_op = func(*arrays)
            arr_results.append(single_op)
        res = ivy.stack(arr_results)

        if out_axes:
            res = tf.experimental.numpy.moveaxis(res, 0, out_axes)

        return res

    return _vmap
