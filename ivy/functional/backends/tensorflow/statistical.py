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
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.math.reduce_max(x, axis=axis, keepdims=keepdims)


def mean(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.math.reduce_mean(x, axis=axis, keepdims=keepdims)


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


def _infer_dtype(dtype: tf.DType):
    default_dtype = ivy.infer_default_dtype(dtype)
    if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
        return default_dtype
    return dtype


def prod(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[tf.DType] = None,
    keepdims: bool = False,
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
    if axis is None:
        axis = tuple(range(len(x.shape)))
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    size = 1
    for a in axis:
        size *= x.shape[a]
    return (size / (size - correction)) ** 0.5 * tf.experimental.numpy.std(
        x, axis=axis, keepdims=keepdims
    )


def sum(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[tf.DType] = None,
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
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if axis is None:
        axis = tuple(range(len(x.shape)))
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size - correction <= 0:
        ret = tf.experimental.numpy.var(x, axis=axis, out=out, keepdims=keepdims)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    else:
        return (size / (size - correction)) ** 0.5 * tf.experimental.numpy.var(
            x, axis=axis, out=out, keepdims=keepdims
        )


# Extra #
# ------#


def cumprod(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = 0,
    exclusive: bool = False,
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
    exclusive: bool = False,
    reverse: bool = False,
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


def einsum(
    equation: str,
    *operands: Union[tf.Tensor, tf.Variable],
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.einsum(equation, *operands)
