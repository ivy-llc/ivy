# global
import tensorflow as tf
from typing import Union, Optional

# local
import ivy


def _cast_for_binary_op(x1, x2):
    return ivy.promote_types_of_inputs(x1, x2)


def _clamp_bits(x1, x2):
    x2 = tf.clip_by_value(
        x2,
        tf.constant(0, dtype=x2.dtype),
        tf.constant(x1.dtype.size * 8 - 1, dtype=x2.dtype),
    )
    return x1, x2


def abs(
    x: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if "uint" in ivy.dtype(x):
        return x
    else:
        return tf.abs(x)


def acos(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.acos(x)


def acosh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.acosh(x)


def add(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.experimental.numpy.add(x1, x2)


def asin(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.asin(x)


def asinh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.asinh(x)


def atan(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.atan(x)


def atan2(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.math.atan2(x1, x2)


def atanh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.atanh(x)


def bitwise_and(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    if ("int" not in str(x1.dtype)) & ("int" not in str(x2.dtype)):
        return tf.math.logical_and(x1, x2)
    else:
        return tf.bitwise.bitwise_and(x1, x2)


def bitwise_invert(
    x: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if "int" not in str(x.dtype):
        return tf.logical_not(x)
    else:
        return tf.bitwise.invert(x)


def bitwise_left_shift(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    x1, x2 = _clamp_bits(x1, x2)
    return tf.bitwise.left_shift(x1, x2)


def bitwise_or(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    if ("int" not in str(x1.dtype)) & ("int" not in str(x2.dtype)):
        return tf.math.logical_or(x1, x2)
    else:
        return tf.bitwise.bitwise_or(x1, x2)


def bitwise_right_shift(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    x1, x2 = _clamp_bits(x1, x2)
    return tf.bitwise.right_shift(x1, x2)


def bitwise_xor(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    if ("int" not in str(x1.dtype)) & ("int" not in str(x2.dtype)):
        return tf.math.logical_xor(x1, x2)
    else:
        return tf.bitwise.bitwise_xor(x1, x2)


def ceil(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if "int" in str(x.dtype):
        return x
    else:
        return tf.math.ceil(x)


def cos(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.cos(x)


def cosh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.cosh(x)


def divide(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    ret = tf.experimental.numpy.divide(x1, x2)
    if ivy.is_float_dtype(x1.dtype):
        ret = tf.cast(ret, dtype=x1.dtype)
    else:
        ret = tf.cast(ret, dtype=ivy.default_float_dtype(as_native=True))
    return ret


def equal(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.math.equal(x1, x2)


def exp(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.exp(x)


def expm1(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.expm1(x)


def floor(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if "int" in str(x.dtype):
        return x
    else:
        return tf.math.floor(x)


def floor_divide(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    ret = tf.experimental.numpy.floor_divide(x1, x2)
    return ret


def greater(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.math.greater(x1, x2)


def greater_equal(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.math.greater_equal(x1, x2)


def isfinite(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if ivy.is_int_dtype(x):
        return tf.ones_like(x, tf.bool)
    else:
        return tf.math.is_finite(x)


def isinf(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if ivy.is_int_dtype(x):
        return tf.zeros_like(x, tf.bool)
    else:
        return tf.math.is_inf(x)


def isnan(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if ivy.is_int_dtype(x):
        return tf.zeros_like(x, tf.bool)
    else:
        return tf.math.is_nan(x)


def less(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.math.less(x1, x2)


def less_equal(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.math.less_equal(x1, x2)


def log(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.log(x)


def log10(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.log(x) / tf.math.log(tf.constant(10.0, x.dtype))


def log1p(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.log1p(x)


def log2(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.log(x) / tf.math.log(tf.constant(2.0, x.dtype))


def logaddexp(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.experimental.numpy.logaddexp(x1, x2)


def logical_and(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.logical_and(tf.cast(x1, tf.bool), tf.cast(x2, tf.bool))


def logical_not(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.logical_not(tf.cast(x, tf.bool))


def logical_or(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.logical_or(tf.cast(x1, tf.bool), tf.cast(x2, tf.bool))


def logical_xor(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.logical_xor(tf.cast(x1, tf.bool), tf.cast(x2, tf.bool))


def multiply(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.math.multiply(x1, x2)


def negative(
    x: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if x.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        return tf.cast(tf.negative(tf.cast(x, tf.float32)), x.dtype)
    return tf.negative(x)


negative.unsupported_dtypes = ("uint8", "uint16", "uint32", "uint64")


def not_equal(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.math.not_equal(x1, x2)


def positive(
    x: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.positive(x)


def pow(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    if isinstance(x1, tf.Tensor) and isinstance(x2, tf.Tensor):
        if x1.dtype.is_unsigned or x2.dtype.is_unsigned:
            promoted_type = tf.experimental.numpy.promote_types(x1.dtype, x2.dtype)
            if x1.dtype.is_unsigned:
                x1 = tf.cast(x1, tf.float64)
            if x2.dtype.is_unsigned:
                x2 = tf.cast(x2, tf.float64)
            return tf.cast(tf.experimental.numpy.power(x1, x2), promoted_type)
    return tf.experimental.numpy.power(x1, x2)


pow.unsupported_dtypes = ("uint8", "uint16", "uint32", "uint64", "float64")


def remainder(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.experimental.numpy.remainder(x1, x2)


def round(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if "int" in str(x.dtype):
        return x
    else:
        return tf.round(x)


def sign(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if x.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        return tf.cast(tf.math.sign(tf.cast(x, tf.float32)), x.dtype)
    return tf.math.sign(x)


def sin(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.sin(x)


def sinh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.sinh(x)


def sqrt(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if x.dtype == "float32":
        x_64 = tf.cast(x, tf.float64)
        return tf.cast(tf.sqrt(x_64), x.dtype)
    else:
        return tf.math.sqrt(x)


def square(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.square(x)


def subtract(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.subtract(x1, x2)


def tan(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.tan(x)


def tanh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.tanh(x)


def trunc(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = x
    if not ivy.is_array(x):
        raise Exception("Input must be array")
    elif not ("int" in str(x.dtype)):
        if not ret.get_shape().ndims == 0:
            ret = tf.tensor_scatter_nd_update(
                x, tf.where(tf.greater(x, 0)), tf.math.floor(x[x > 0])
            )
            ret = tf.tensor_scatter_nd_update(
                ret, tf.where(tf.less(x, 0)), tf.math.ceil(x[x < 0])
            )
        else:
            ret = (tf.math.floor if ret > 0 else tf.math.ceil)(ret)
    return ret


# Extra #
# ------#


def erf(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.erf(x)


def maximum(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.maximum(x1, x2)


def minimum(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return tf.minimum(x1, x2)
