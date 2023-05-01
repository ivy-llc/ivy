# global
_round = round
import tensorflow as tf
from typing import Union, Optional, Sequence, Tuple
# local
import ivy
from ivy.functional.ivy.statistical import _get_promoted_type_of_operands
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


# Array API Standard #
# -------------------#


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def min(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.math.reduce_min(x, axis=axis, keepdims=keepdims)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def max(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
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


def _infer_dtype(dtype: tf.DType):
    default_dtype = ivy.infer_default_dtype(dtype)
    if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
        return default_dtype
    return dtype


def prod(
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
    return tf.experimental.numpy.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)


def std(
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
        ret = tf.experimental.numpy.std(x, axis=axis, keepdims=keepdims)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    else:
        return tf.cast(
            tf.math.multiply(
                tf.experimental.numpy.std(x, axis=axis, keepdims=keepdims),
                (size / (size - correction)) ** 0.5,
            ),
            x.dtype,
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
    if dtype is None and not ivy.is_bool_dtype(x):
        dtype = x.dtype
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
    if correction == 0:
        return tf.experimental.numpy.var(x, axis=axis, out=out, keepdims=keepdims)
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size - correction <= 0:
        ret = tf.experimental.numpy.var(x, axis=axis, out=out, keepdims=keepdims)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    else:
        return ivy.astype(
            tf.math.multiply(
                tf.experimental.numpy.var(x, axis=axis, out=out, keepdims=keepdims),
                size / (size - correction),
            ),
            x.dtype,
            copy=False,
        )


# Extra #
# ------#


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def cumprod(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        if dtype is tf.bool:
            dtype = ivy.default_int_dtype()
        else:
            dtype = _infer_dtype(x.dtype)
        dtype = ivy.as_native_dtype(dtype)
    x = tf.cast(x, dtype)
    return tf.math.cumprod(x, axis, exclusive, reverse)


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
        elif ivy.is_int_dtype(x.dtype):
            dtype = ivy.promote_types(x.dtype, ivy.default_int_dtype(as_native=True))
        else:
            dtype = _infer_dtype(x.dtype)
        dtype = ivy.as_native_dtype(dtype)
    x = tf.cast(x, dtype)
    return tf.math.cumsum(x, axis, exclusive, reverse)


def cummax(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    if x.dtype == tf.bool:
        x = tf.cast(x, tf.float64)
    elif x.dtype == tf.int16 or x.dtype == tf.int8:
        x = tf.cast(x, tf.int64)
    elif x.dtype == tf.complex128 or x.dtype == tf.complex64:
        x = tf.math.real(x)
    if exclusive or reverse:
        if exclusive and reverse:
            x, indices = __find_cummax(tf.experimental.numpy.flip(x, axis=axis),
                                       axis=axis)
            x, indices = tf.experimental.numpy.swapaxes(x, axis, -1), \
                tf.experimental.numpy.swapaxes(indices, axis, -1)
            x, indices = tf.experimental.numpy.concatenate((
                tf.experimental.numpy.zeros_like(x[..., -1:]), x[..., :-1]), -1), \
                tf.experimental.numpy.concatenate((
                    tf.experimental.numpy.zeros_like(indices[..., -1:]),
                    indices[..., :-1]), -1)
            x, indices = tf.experimental.numpy.swapaxes(x, axis, -1), \
                tf.experimental.numpy.swapaxes(indices, axis, -1)
            res, indices = tf.experimental.numpy.flip(x, axis=axis), \
                tf.experimental.numpy.flip(indices, axis=axis)
        elif exclusive:
            x = tf.experimental.numpy.swapaxes(x, axis, -1)
            x = tf.experimental.numpy.concatenate((
                tf.experimental.numpy.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = tf.experimental.numpy.swapaxes(x, axis, -1)
            res, indices = __find_cummax(x, axis=axis)
        elif reverse:
            x = tf.experimental.numpy.flip(x, axis=axis)
            x, indices = __find_cummax(x, axis=axis)
            res, indices = tf.experimental.numpy.flip(x, axis=axis), \
                tf.experimental.numpy.flip(indices, axis=axis)
        return res, indices

    return __find_cummax(x, axis=axis)


def __find_cummax(
        x: tf.Tensor,
        axis: int = 0
) -> Tuple[tf.Tensor, tf.Tensor]:
    indices = []
    values = []
    if isinstance(x[0], tf.Tensor) and isinstance(x[0].numpy().tolist(), list) \
            and len(x[0].numpy().tolist()) >= 1 :
        if axis == 1:
            for ret1 in x:
                ret1_indices = tf.convert_to_tensor(list(range(0, ret1.shape[0])),
                                                    dtype=ret1.dtype)
                value, indice = tf.scan(lambda a, b: a if a > b
                                        or tf.experimental.numpy.where
                                        (ret1[0].numpy() == b[0].numpy()) == 0
                                        else b, (ret1, ret1_indices))
                indices.append(indice)
                values.append(value)
        else:
            n1 = {}
            indices, values = [] , []
            x_list = x.numpy().tolist()
            for index, ret1 in enumerate(x_list):
                indice = []
                value = []
                for idx1, x1 in enumerate(ret1):
                    if index == 0 or x_list[index][idx1] >= x_list[n1[idx1]][idx1]:
                        n1[idx1] = index
                        indice.append(index)
                        value.append(x_list[index][idx1])
                    else:
                        indice.append(n1[idx1])
                        value.append(x_list[n1[idx1]][idx1])
                indices.append(indice)
                values.append(value)
    else:
        x_indices = tf.convert_to_tensor(list(range(0, x.shape[0])), dtype=x.dtype)
        values, indices = tf.scan(lambda a, b: a if a > b
                                  or tf.experimental.numpy.where
                                  (x[0].numpy() == b[0].numpy()) == 0
                                  else b, (x, x_indices))

    return tf.convert_to_tensor(values, dtype=x.dtype), \
        tf.cast(tf.convert_to_tensor(indices), dtype=tf.int64)


def einsum(
    equation: str,
    *operands: Union[tf.Tensor, tf.Variable],
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = _get_promoted_type_of_operands(operands)
    operands = (tf.cast(operand, tf.float32) for operand in operands)
    return tf.cast(tf.einsum(equation, *operands), dtype)
