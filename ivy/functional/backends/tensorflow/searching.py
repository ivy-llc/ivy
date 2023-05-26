# global
from numbers import Number
from typing import Optional, Union, Tuple

import tensorflow as tf

import ivy


# Array API Standard #
# ------------------ #


def argmax(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    select_last_index: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if select_last_index:
        x = tf.experimental.numpy.flip(x, axis=axis)
        ret = x.numpy().argmax(axis=axis, keepdims=keepdims)
        if axis is not None:
            ret = x.shape[axis] - ret - 1
        else:
            ret = tf.size(x, out_type=tf.int64) - ret - 1
    else:
        ret = x.numpy().argmax(axis=axis, keepdims=keepdims)
    if dtype is not None:
        dtype = ivy.as_native_dtype(dtype)
        return tf.cast(ret, dtype)
    return tf.convert_to_tensor(ret)


def argmin(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[tf.dtypes.DType] = None,
    select_last_index: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if select_last_index:
        x = tf.experimental.numpy.flip(x, axis=axis)
        ret = x.numpy().argmin(axis=axis, keepdims=keepdims)
        if axis is not None:
            ret = x.shape[axis] - ret - 1
        else:
            ret = tf.size(x, out_type=tf.dtypes.int64) - ret - 1
    else:
        ret = x.numpy().argmin(axis=axis, keepdims=keepdims)
    if output_dtype is not None:
        output_dtype = ivy.as_native_dtype(output_dtype)
        return tf.cast(ret, output_dtype)
    return tf.convert_to_tensor(ret)


def nonzero(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[tf.Tensor, tf.Variable, Tuple[Union[tf.Tensor, tf.Variable]]]:
    res = tf.experimental.numpy.nonzero(x)

    if size is not None:
        dtype = tf.int64
        if isinstance(fill_value, float):
            dtype = tf.float64
        res = tf.cast(res, dtype)

        diff = size - res[0].shape[0]
        if diff > 0:
            res = tf.pad(res, [[0, 0], [0, diff]], constant_values=fill_value)
        elif diff < 0:
            res = tf.slice(res, [0, 0], [-1, size])

    if as_tuple:
        return tuple(res)
    return tf.stack(res, axis=1)


def where(
    condition: Union[tf.Tensor, tf.Variable],
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.cast(tf.experimental.numpy.where(condition, x1, x2), x1.dtype)


# Extra #
# ----- #


def argwhere(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if x.ndim == 0:
        return tf.zeros(shape=[int(bool(x)), 0], dtype="int64")
    where_x = tf.experimental.numpy.nonzero(x)
    res = tf.experimental.numpy.concatenate(
        [tf.expand_dims(item, -1) for item in where_x], -1
    )
    return res
