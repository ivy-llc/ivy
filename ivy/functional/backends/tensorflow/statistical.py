# global
import tensorflow as tf
from typing import Union, Optional, Sequence

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from ivy.utils.einsum_parser import legalise_einsum_expr

# Array API Standard #
# -------------------#


@with_unsupported_dtypes(
    {"2.15.0 and below": ("complex", "bool", "uint64")}, backend_version
)
def min(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float, complex]] = None,
    where: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    if where is not None:
        max_val = (
            ivy.iinfo(x.dtype).max
            if ivy.is_int_dtype(x.dtype)
            else ivy.finfo(x.dtype).max
        )
        x = tf.where(where, x, tf.ones_like(x) * max_val)
    result = tf.math.reduce_min(x, axis=axis, keepdims=keepdims)
    if initial is not None:
        result = tf.minimum(result, initial)
    return result


@with_unsupported_dtypes({"2.15.0 and below": ("bool",)}, backend_version)
def max(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if "complex" in str(x.dtype):
        real = tf.math.real(x)
        img = tf.math.imag(x)
        const = tf.constant(1j, dtype=x.dtype)
        real_max = tf.reduce_max(real, axis=axis, keepdims=keepdims)
        imag = tf.where(
            real == real_max,
            img,
            tf.experimental.numpy.finfo(img.dtype).min,
        )
        # we consider the number with the biggest real and imag part
        img_max = tf.reduce_max(imag, axis=axis, keepdims=keepdims)
        img_max = tf.cast(img_max, x.dtype)
        return tf.add(tf.cast(real_max, x.dtype), tf.multiply(img_max, const))
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tf.math.reduce_max(x, axis=axis, keepdims=keepdims)


@with_unsupported_dtypes({"2.15.0 and below": ("bool",)}, backend_version)
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
    keepdims: Optional[bool] = False,
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
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size - correction <= 0:
        ret = tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)
        ret = tf.cast(tf.fill(ret.shape, float("nan")), ret.dtype)
        return ret
    else:
        return (
            tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)
            * size
            / (size - correction)
        )


# Extra #
# ------#


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "bool")}, backend_version)
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


@with_unsupported_dtypes({"2.15.0 and below": "bool"}, backend_version)
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


@with_unsupported_dtypes(
    {"2.15.0 and below": ("unsigned", "int8", "int16")},
    backend_version,
)
def einsum(
    equation: str,
    *operands: Union[tf.Tensor, tf.Variable],
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    equation = legalise_einsum_expr(*[equation, *operands])
    dtype_list = set(map(lambda x: x.dtype, operands))
    dtype = dtype_list.pop()
    if len(dtype_list) > 0:
        for d in dtype_list:
            dtype = ivy.promote_types(dtype, d)
        dtype = ivy.as_native_dtype(dtype)
        operands = list(
            map(lambda x: tf.cast(x, dtype) if x.dtype != dtype else x, operands)
        )

    return tf.einsum(equation, *operands)
