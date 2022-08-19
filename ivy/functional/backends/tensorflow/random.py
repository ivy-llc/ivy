"""Collection of TensorFlow random functions, wrapped to fit Ivy syntax and
signature.
"""

# global
import tensorflow as tf
from tensorflow.python.framework.dtypes import DType
from typing import Optional, Union, Sequence

# local
import ivy
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)

# Extra #
# ------#


def random_uniform(
    *,
    low: Union[float, tf.Tensor, tf.Variable] = 0.0,
    high: Union[float, tf.Tensor, tf.Variable] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    shape = _check_bounds_and_get_shape(low, high, shape)
    low = tf.cast(low, dtype)
    high = tf.cast(high, dtype)
    with tf.device(device):
        return tf.random.uniform(shape, low, high, dtype=dtype)


def random_normal(
    *,
    mean: Union[float, tf.Tensor, tf.Variable] = 0.0,
    std: Union[float, tf.Tensor, tf.Variable] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape)
    mean = tf.cast(mean, dtype)
    std = tf.cast(std, dtype)
    with tf.device(device):
        return tf.random.normal(shape, mean, std, dtype=dtype)


def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[Union[tf.Tensor, tf.Variable]] = None,
    replace: bool = True,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not replace:
        raise Exception("TensorFlow does not support multinomial without replacement")
    with tf.device(device):
        if probs is None:
            probs = (
                tf.ones(
                    (
                        batch_size,
                        population_size,
                    )
                )
                / population_size
            )
        return tf.random.categorical(tf.math.log(probs), num_samples)


def randint(
    low: Union[float, tf.Tensor, tf.Variable],
    high: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str,
    dtype: Optional[Union[DType, ivy.Dtype]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape)
    low = tf.cast(low, "float32")
    high = tf.cast(high, "float32")
    with tf.device(device):
        return tf.cast(tf.random.uniform(shape, low, high, "float32"), dtype)


def seed(*, seed_value: int = 0) -> None:
    tf.random.set_seed(seed_value)


def shuffle(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.random.shuffle(x)
