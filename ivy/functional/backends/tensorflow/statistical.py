# global
_round = round
import tensorflow as tf
from typing import Tuple, Union, Optional

# local
import ivy


# Array API Standard #
# -------------------#


def max(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.reduce_max(x, axis=axis, keepdims=keepdims)


def mean(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Union[tf.Tensor, tf.Variable]:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def min(
    x: Union[tf.Tensor, tf.Variable],
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.math.reduce_min(x, axis=axis, keepdims=keepdims)


def prod(
    x: Union[tf.Tensor, tf.Variable],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: tf.DType = None,
    keepdims: bool = False,
) -> Union[tf.Tensor, tf.Variable]:
    if dtype is None:
        if x.dtype in [tf.int8, tf.int16, tf.int32]:
            dtype = tf.int32
        elif x.dtype in [tf.uint8, tf.uint16, tf.experimental.numpy.uint32]:
            dtype = tf.experimental.numpy.uint32
        elif x.dtype == tf.int64:
            dtype = tf.int64
        elif x.dtype == tf.uint64:
            dtype = tf.uint64
    dtype = ivy.as_native_dtype(dtype)
    return tf.experimental.numpy.prod(x, axis, dtype, keepdims)


def std(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.std(x, axis, keepdims)


def sum(
    x: Union[tf.Tensor, tf.Variable],
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: tf.DType = None,
    keepdims: bool = False,
) -> Union[tf.Tensor, tf.Variable]:
    if dtype is None:
        if x.dtype in [tf.int8, tf.int16, tf.int32]:
            dtype = tf.int32
        elif x.dtype in [tf.uint8, tf.uint16, tf.experimental.numpy.uint32]:
            dtype = tf.experimental.numpy.uint32
        elif x.dtype == tf.int64:
            dtype = tf.int64
        elif x.dtype == tf.uint64:
            dtype = tf.uint64
    dtype = ivy.as_native_dtype(dtype)
    return tf.experimental.numpy.sum(x, axis, dtype, keepdims)


def var(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)


# Extra #
# ------#


def einsum(
    equation: str,
    *operands: Union[tf.Tensor, tf.Variable],
) -> Union[tf.Tensor, tf.Variable]:
    return tf.einsum(equation, *operands)
