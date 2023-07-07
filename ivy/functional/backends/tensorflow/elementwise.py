# global
from typing import Union, Optional
import tensorflow as tf
import tensorflow_probability as tfp

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy import promote_types_of_inputs
from . import backend_version


def abs(
    x: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    where: Union[bool, tf.Tensor, tf.Variable] = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x_dtype = ivy.dtype(x)
    if any(("uint" in x_dtype, "bool" in x_dtype)):
        return x
    ret = ivy.where(where, tf.abs(x), x)
    if ivy.is_complex_dtype(x_dtype):
        return ivy.real(ret)
    return ret


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
        with ivy.ArrayMode(False):
            x2 = multiply(x2, alpha)
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


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def bitwise_left_shift(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return tf.bitwise.left_shift(x1, x2)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def bitwise_right_shift(
    x1: Union[int, tf.Tensor, tf.Variable],
    x2: Union[int, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return tf.bitwise.right_shift(x1, x2)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("float16",)}, backend_version)
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
    if ivy.is_float_dtype(x1.dtype) or ivy.is_complex_dtype(x1.dtype):
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


def exp2(
    x: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.pow(2, x, name=None)


def expm1(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.expm1(x)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def floor_divide(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.experimental.numpy.floor_divide(x1, x2)


@with_supported_dtypes({"2.13.0 and below": ("float",)}, backend_version)
def fmin(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = promote_types_of_inputs(x1, x2)
    x1 = tf.where(tf.math.is_nan(x1), x2, x1)
    x2 = tf.where(tf.math.is_nan(x2), x1, x2)
    ret = tf.experimental.numpy.minimum(x1, x2)
    return ret


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def greater(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.math.greater(x1, x2)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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
    elif ivy.is_complex_dtype(x):
        return tf.math.logical_and(
            tf.math.is_finite(tf.math.real(x)), tf.math.is_finite(tf.math.imag(x))
        )
    else:
        return tf.math.is_finite(x)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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
            return tf.experimental.numpy.isneginf(x)
        elif detect_positive:
            return tf.experimental.numpy.isposinf(x)
        return tf.zeros_like(x, tf.bool)


@with_unsupported_dtypes({"2.13.0 and below": ("complex", "bool")}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("unsigned",)}, backend_version)
def lcm(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return tf.experimental.numpy.lcm(x1, x2)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def less(
    x1: Union[float, tf.Tensor, tf.Variable],
    x2: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.math.less(x1, x2)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("float16", "bfloat16")}, backend_version)
def logaddexp(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.experimental.numpy.logaddexp(x1, x2)


@with_unsupported_dtypes({"2.13.0 and below": ("float16",)}, backend_version)
def real(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.real(x)


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
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
def logaddexp2(
    x1: Union[tf.Tensor, tf.Variable, float, list, tuple],
    x2: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not ivy.is_float_dtype(x1):
        x1 = tf.cast(x1, ivy.default_float_dtype(as_native=True))
        x2 = tf.cast(x2, ivy.default_float_dtype(as_native=True))
    amax = ivy.maximum(x1, x2)
    delta = x1 - x2
    return ivy.where(
        ivy.isnan(delta),
        x1 + x2,
        amax + ivy.log1p(ivy.exp2(-ivy.abs(delta))) / ivy.log(2.0).astype(amax.dtype),
    )


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
    {"2.13.0 and below": ("uint8", "uint16", "uint32", "uint64")}, backend_version
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
    {"2.13.0 and below": ("uint8", "uint16", "uint32", "uint64", "float64")},
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


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16", "complex")}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16", "complex")}, backend_version)
def round(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    decimals: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if "int" in str(x.dtype):
        return x
    else:
        if decimals == 0:
            return tf.cast(tf.round(x), x.dtype)
        ret_dtype = x.dtype
        factor = tf.constant(10**decimals, dtype=ret_dtype)
        factor_deno = tf.where(
            tf.math.is_finite(factor), factor, tf.constant(1, dtype=ret_dtype)
        )
        return tf.cast(tf.round(x * factor) / factor_deno, ret_dtype)


def sign(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    np_variant: Optional[bool] = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if x.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        return tf.cast(tf.math.sign(tf.cast(x, tf.float32)), x.dtype)
    if x.dtype in [tf.complex64, tf.complex128] and np_variant:
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        return tf.cast(tf.where(real != 0, tf.sign(real), tf.sign(imag)), x.dtype)
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


def trapz(
    y: Union[tf.Tensor, tf.Variable],
    /,
    *,
    x: Optional[Union[tf.Tensor, tf.Variable]] = None,
    dx: float = 1.0,
    axis: int = -1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tfp.math.trapz(y, x=x, dx=dx, axis=axis, name=None)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
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


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def erf(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.erf(x)


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
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
        "2.13.0 and below": (
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
        "2.13.0 and below": (
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


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, backend_version)
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


@with_unsupported_dtypes(
    {"2.13.0 and below": ("uint8", "uint16", "uint32", "uint64", "complex", "bool")},
    backend_version,
)
def fmod(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = promote_types_of_inputs(x1, x2)
    # tf.math.floormod returns wrong results
    res = tf.experimental.numpy.remainder(tf.math.abs(x1), tf.math.abs(x2))
    return tf.where(x1 < 0, -res, res)


@with_unsupported_dtypes(
    {"2.13.0 and below": ("uint8", "uint16", "uint32", "uint64")}, backend_version
)
def gcd(
    x1: Union[tf.Tensor, tf.Variable, int, list, tuple],
    x2: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return tf.experimental.numpy.gcd(x1, x2)


gcd.support_native_out = False


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "bfloat16",
            "int32",
        )
    },
    backend_version,
)
def angle(
    input: Union[tf.Tensor, tf.Variable],
    /,
    *,
    deg: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if deg:
        return tf.math.angle(input, name=None) * (180 / tf.experimental.numpy.pi)
    else:
        return tf.math.angle(input, name=None)


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "bfloat16",
            "int32",
        )
    },
    backend_version,
)
def imag(
    val: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.imag(val, name=None)


def nan_to_num(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: bool = True,
    nan: Union[float, int] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    posinf = posinf if posinf is not None else x.dtype.max
    neginf = neginf if neginf is not None else x.dtype.min
    posinf = tf.constant(posinf, x.dtype)
    neginf = tf.constant(neginf, x.dtype)
    nan = tf.constant(nan, x.dtype)
    ret = tf.where(tf.math.is_nan(x), nan, x)
    ret = tf.where(tf.math.logical_and(tf.math.is_inf(ret), ret > 0), posinf, ret)
    ret = tf.where(tf.math.logical_and(tf.math.is_inf(ret), ret < 0), neginf, ret)
    if copy:
        return ret
    else:
        x = ret
        return x
