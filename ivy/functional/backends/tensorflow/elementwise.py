# global
from typing import Union, Optional
import tensorflow as tf

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def abs(
    x: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
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
    alpha: Optional[Union[int, float]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        ivy.set_array_mode(False)
        x2 = multiply(x2, alpha)
        ivy.unset_array_mode()
    return tf.add(x1, x2)


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


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def atan2(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.math.atan2(x1, x2)


def atanh(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.atanh(x)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def bitwise_and(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    if ("int" not in str(x1.dtype)) & ("int" not in str(x2.dtype)):
        return tf.math.logical_and(x1, x2)
    else:
        return tf.bitwise.bitwise_and(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def bitwise_left_shift(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return tf.bitwise.left_shift(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def bitwise_or(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    if ("int" not in str(x1.dtype)) & ("int" not in str(x2.dtype)):
        return tf.math.logical_or(x1, x2)
    else:
        return tf.bitwise.bitwise_or(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def bitwise_right_shift(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return tf.bitwise.right_shift(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def bitwise_xor(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    if ("int" not in str(x1.dtype)) & ("int" not in str(x2.dtype)):
        return tf.math.logical_xor(x1, x2)
    else:
        return tf.bitwise.bitwise_xor(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.9.1 and below": ("float16",)}, backend_version)
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
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
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
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
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


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def floor_divide(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.experimental.numpy.floor_divide(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def greater(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.math.greater(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def greater_equal(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
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


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def isinf(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    detect_positive: bool = True,
    detect_negative: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if ivy.is_int_dtype(x):
        return tf.zeros_like(x, tf.bool)
    else:
        if detect_negative and detect_positive:
            return tf.math.is_inf(x)
        elif detect_negative:
            return tf.experimental.numpy.isposinf(x)
        elif detect_positive:
            return tf.experimental.numpy.isneginf(x)
        return tf.zeros_like(x, tf.bool)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def less(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.math.less(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def less_equal(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
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


@with_unsupported_dtypes({"2.9.1 and below": ("float16", "bfloat16")}, backend_version)
def logaddexp(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    # ToDo: implement using tf.experimental.numpy.logaddexp if this becomes stable and
    # supports gradients in future
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    dtype = x1.dtype
    x1 = tf.cast(x1, tf.float64)
    x2 = tf.cast(x2, tf.float64)
    return ivy.log(ivy.add(ivy.exp(x1), ivy.exp(x2))).astype(dtype)


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
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.math.multiply(x1, x2)


@with_unsupported_dtypes(
    {"2.9.1 and below": ("uint8", "uint16", "uint32", "uint64")}, backend_version
)
def negative(
    x: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if x.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        return tf.cast(tf.negative(tf.cast(x, tf.float32)), x.dtype)
    return tf.negative(x)


def not_equal(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.math.not_equal(x1, x2)


def positive(
    x: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.positive(x)


@with_unsupported_dtypes(
    {"2.9.1 and below": ("uint8", "uint16", "uint32", "uint64", "float64")},
    backend_version,
)
def pow(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if isinstance(x1, tf.Tensor) and isinstance(x2, tf.Tensor):
        if x1.dtype.is_unsigned or x2.dtype.is_unsigned:
            promoted_type = tf.experimental.numpy.promote_types(x1.dtype, x2.dtype)
            if x1.dtype.is_unsigned:
                x1 = tf.cast(x1, tf.float64)
            if x2.dtype.is_unsigned:
                x2 = tf.cast(x2, tf.float64)
            return tf.cast(tf.experimental.numpy.power(x1, x2), promoted_type)
    return tf.experimental.numpy.power(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16", "complex")}, backend_version)
def remainder(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    modulus: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = tf.where(res >= 0, tf.math.floor(res), tf.math.ceil(res))
        diff = res - res_floored
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        return tf.cast(tf.round(diff * x2), x1.dtype)
    return tf.experimental.numpy.remainder(x1, x2)


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16", "complex")}, backend_version)
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
    alpha: Optional[Union[int, float]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        ivy.set_array_mode(False)
        x2 = multiply(x2, alpha)
        ivy.unset_array_mode()
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


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def trunc(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = x
    if not ivy.is_array(x):
        raise ivy.utils.exceptions.IvyException("Input must be array")
    elif not ("int" in str(x.dtype)):
        if not ret.get_shape().ndims == 0:
            ret = tf.tensor_scatter_nd_update(
                x, tf.where(tf.greater_equal(x, 0)), tf.math.floor(x[x >= 0])
            )
            ret = tf.tensor_scatter_nd_update(
                ret, tf.where(tf.less(x, 0)), tf.math.ceil(x[x < 0])
            )
        else:
            ret = (tf.math.floor if ret >= 0 else tf.math.ceil)(ret)
    return ret


# Extra #
# ------#


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def erf(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.erf(x)


@with_unsupported_dtypes(
    {
        "2.9.1 and below": (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "complex",
        )
    },
    backend_version,
)
def maximum(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    use_where: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    dtype = x1.dtype
    if use_where:
        return tf.math.maximum(x1, x2)
    x1 = tf.cast(x1, tf.float64)
    x2 = tf.cast(x2, tf.float64)
    return tf.cast((x1 + x2 + tf.math.abs(x1 - x2)) / 2, dtype=dtype)


@with_unsupported_dtypes(
    {
        "2.9.1 and below": (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "complex",
        )
    },
    backend_version,
)
def minimum(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    use_where: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    dtype = x1.dtype
    if use_where:
        return tf.math.minimum(x1, x2)
    x1 = tf.cast(x1, tf.float64)
    x2 = tf.cast(x2, tf.float64)
    return tf.cast((x1 + x2 - tf.math.abs(x1 - x2)) / 2, dtype)


@with_unsupported_dtypes(
    {
        "2.9.1 and below": (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    backend_version,
)
def reciprocal(
    x: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.reciprocal(x)


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16",)}, backend_version)
def deg2rad(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.deg2rad(x)


def rad2deg(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.rad2deg(x)


def isreal(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.isreal(x)
