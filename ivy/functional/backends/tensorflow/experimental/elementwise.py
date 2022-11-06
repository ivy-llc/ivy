from typing import Union, Optional
import tensorflow as tf
from .. import backend_version


# local
from ivy.func_wrapper import with_unsupported_dtypes
import tensorflow_probability as tfp


def sinc(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    return tf.cast(tf.experimental.numpy.sinc(x), x.dtype)


@with_unsupported_dtypes(
    {"2.9.1 and below": ("uint8", "uint16", "uint32", "uint64")}, backend_version
)
def lcm(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if [x1.dtype, x2.dtype] == [tf.int8, tf.int8]:
        dtype = tf.int8
        x1 = tf.cast(x1, dtype=tf.int16)
        x2 = tf.cast(x2, dtype=tf.int16)
    else:
        dtype = x1.dtype
    return tf.math.abs(tf.cast(tf.experimental.numpy.lcm(x1, x2), dtype=dtype))


def fmod(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.floormod(x1, x2, name=None)


@with_unsupported_dtypes(
    {"2.9.1 and below": ("blfoat16", "float16", "float32", "float64")}, backend_version
)
def fmax(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1 = tf.where(tf.math.is_nan(x1), float("inf"), x1)
    x2 = tf.where(tf.math.is_nan(x1), float("inf"), x2)
    ret = tf.math.maximum(x1, x2, name=None)
    return tf.where(tf.math.is_inf(ret), float("nan"))


def trapz(
    y: Union[tf.Tensor, tf.Variable],
    /,
    *,
    x: Optional[Union[tf.Tensor, tf.Variable]] = None,
    dx: Optional[float] = 1.0,
    axis: Optional[int] = -1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tfp.math.trapz(y, x=x, dx=dx, axis=axis, name=None)


def float_power(
    x1: Union[tf.Tensor, tf.Variable, float, list, tuple],
    x2: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.float_power(x1, x2)


def exp2(
    x: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.pow(2, x, name=None)


def nansum(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[tuple, int]] = None,
    dtype: Optional[tf.DType] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.nansum(x, axis=axis, dtype=dtype, keepdims=keepdims)


@with_unsupported_dtypes(
    {"2.9.1 and below": ("uint8", "uint16", "uint32", "uint64")}, backend_version
)
def gcd(
    x1: Union[tf.Tensor, tf.Variable, int, list, tuple],
    x2: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.gcd(x1, x2)


def isclose(
    a: Union[tf.Tensor, tf.Variable],
    b: Union[tf.Tensor, tf.Variable],
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.isclose(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def isposinf(
    x: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.isposinf(x)


def isneginf(
    x: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.isneginf(x)


def nan_to_num(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = True,
    nan: Optional[Union[float, int]] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    posinf = posinf if posinf is not None else 1.79769313e+308
    neginf = neginf if neginf is not None else -1.79769313e+308
    ret = tf.where(tf.math.is_nan(x), nan, x)
    ret = tf.where(tf.math.logical_and(tf.math.is_inf(ret), ret > 0), posinf, ret)
    ret = tf.where(tf.math.logical_and(tf.math.is_inf(ret), ret < 0), neginf, ret)
    if copy:
        return ret
    else:
        x = ret
        return x

    
@with_unsupported_dtypes(
    {"2.9.1 and below": ("uint8", "uint16", "uint32", "uint64",
                         "int8", "int16", "int32", "int64",)},
    backend_version
)
def logaddexp2(
    x1: Union[tf.Tensor, tf.Variable, float, list, tuple],
    x2: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x = 2**x1 + 2**x2
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator
