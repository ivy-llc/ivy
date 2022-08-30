# global
_round = round
import tensorflow as tf
from typing import Tuple, Union, Optional, Sequence

# local
import ivy


def _new_var_fun(x, *, axis, correction, dtype):
    output = tf.cast(x, dtype)
    length = tf.cast(tf.shape(output)[axis], dtype)
    divisor = tf.cast(length - correction, dtype)
    mean = tf.math.reduce_sum(output, axis=axis) / length
    output = tf.math.abs(
        tf.cast(output, dtype=dtype) - tf.cast(tf.expand_dims(mean, axis), dtype=dtype)
    )
    output = output**2
    output = tf.math.reduce_sum(output, axis=axis) / divisor
    return output


def _new_std_fun(x, *, axis, correction, dtype):
    return tf.math.sqrt(_new_var_fun(x, axis=axis, correction=correction, dtype=dtype))


# Array API Standard #
# -------------------#


def max(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.math.reduce_max(x, axis=axis, keepdims=keepdims)


def mean(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def min(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.math.reduce_min(x, axis=axis, keepdims=keepdims)


def _infer_dtype(x_dtype: tf.DType):
    default_dtype = ivy.infer_default_dtype(x_dtype)
    if ivy.dtype_bits(x_dtype) < ivy.dtype_bits(default_dtype):
        dtype = default_dtype
    else:
        dtype = x_dtype
    return dtype


def prod(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[tf.DType] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.experimental.numpy.prod(x, axis, dtype, keepdims)


def std(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    dtype = x.dtype
    if isinstance(axis, tuple):
        ret = []
        for i in axis:
            ret.append(
                _new_std_fun(x, axis=i, correction=correction, dtype=dtype).numpy()
            )
        ret = tf.constant(ret, dtype=dtype)
    elif isinstance(axis, int):
        ret = _new_std_fun(x, axis=axis, correction=correction, dtype=dtype)
    else:
        size = tf.size(x).numpy()
        ret = _new_std_fun(
            tf.reshape(x, size), axis=0, correction=correction, dtype=dtype
        )

    if keepdims:
        ret = tf.constant(ret, shape=x.shape)
    return ret


def sum(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: tf.DType = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.experimental.numpy.sum(x, axis, dtype, keepdims)


def var(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if axis is None:
        axis = tuple(range(len(x.shape)))
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    size = 1
    for a in axis:
        size *= x.shape[a]
    return (size / (size - correction)) * tf.experimental.numpy.var(
        x, axis=axis, out=out, keepdims=keepdims
    )


# Extra #
# ------#


def einsum(
    equation: str,
    *operands: Union[tf.Tensor, tf.Variable],
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.einsum(equation, *operands)
