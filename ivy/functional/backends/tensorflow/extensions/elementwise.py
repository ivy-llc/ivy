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
