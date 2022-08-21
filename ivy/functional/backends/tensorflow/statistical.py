# global
_round = round
import tensorflow as tf
from typing import Tuple, Union, Optional, Sequence

# local
import ivy


# Array API Standard #
# -------------------#


def max(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.reduce_max(x, axis=axis, keepdims=keepdims)


def mean(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def min(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.reduce_min(x, axis=axis, keepdims=keepdims)


def prod(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[tf.DType] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
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
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
<<<<<<< HEAD
    return tf.experimental.numpy.std(x, axis, keepdims)
=======
    axis = tuple(axis) if isinstance(axis, list) else axis
    dtype = x.dtype
    if isinstance(axis, tuple):
        ret = []
        for i in axis:
            ret.append(_new_std_fun(
                x,
                axis=i,
                correction=correction,
                dtype=dtype).numpy()
            )
        ret = tf.constant(ret, dtype=dtype)
    elif isinstance(axis, int):
        ret = _new_std_fun(x, axis=axis, correction=correction, dtype=dtype)
    else:
        size = tf.size(x).numpy()
        ret = _new_std_fun(
            tf.reshape(x, size),
            axis=0,
            correction=correction,
            dtype=dtype
        )

    if keepdims:
        ret = tf.constant(ret, shape=x.shape)
    return ret
>>>>>>> 241a3c87d774fb0877df3ef70ff67e83a6cbe4be


def sum(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: tf.DType = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
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
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
<<<<<<< HEAD
    return tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)
=======
    axis = tuple(axis) if isinstance(axis, list) else axis
    dtype = x.dtype
    if isinstance(axis, tuple):
        ret = []
        for i in axis:
            ret.append(_new_var_fun(
                x,
                axis=i,
                correction=correction,
                dtype=dtype).numpy()
            )
        ret = tf.constant(ret, dtype=dtype)
    elif isinstance(axis, int):
        ret = _new_var_fun(
            x,
            axis=axis,
            correction=correction,
            dtype=dtype
        )
    else:
        size = tf.size(x).numpy()
        ret = _new_var_fun(
            tf.reshape(x, size),
            axis=0,
            correction=correction,
            dtype=dtype
        )

    if keepdims:
        ret = tf.constant(ret, shape=x.shape)
    return ret
>>>>>>> 241a3c87d774fb0877df3ef70ff67e83a6cbe4be


# Extra #
# ------#


def einsum(
    equation: str,
    *operands: Union[tf.Tensor, tf.Variable],
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.einsum(equation, *operands)
