"""Collection of TensorFlow random functions, wrapped to fit Ivy syntax and
signature.
"""

# global
import tensorflow as tf
from tensorflow.python.framework.dtypes import DType
from typing import Optional, Union, Sequence

# local
import ivy
from ivy.functional.ivy.random import _check_bounds_and_get_shape

# Extra #
# ------#


def random_uniform(
    low: Union[float, tf.Tensor, tf.Variable] = 0.0,
    high: Union[float, tf.Tensor, tf.Variable] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    dtype: DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    shape = _check_bounds_and_get_shape(low, high, shape)
    low = tf.cast(low, dtype)
    high = tf.cast(high, dtype)
    with tf.device(device):
        return tf.random.uniform(shape, low, high, dtype=dtype)


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    dtype: DType,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    mean = tf.cast(mean, "float32")
    std = tf.cast(std, "float32")
    with tf.device(device):
        return tf.random.normal(shape if shape else (), mean, std)


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[Union[tf.Tensor, tf.Variable]] = None,
    replace: bool = True,
    *,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
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
    low: int,
    high: int,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    low = tf.cast(low, "int64")
    high = tf.cast(high, "int64")
    with tf.device(device):
        return tf.random.uniform(shape=shape, minval=low, maxval=high, dtype=tf.int64)


def seed(seed_value: int = 0) -> None:
    tf.random.set_seed(seed_value)


def shuffle(
    x: Union[tf.Tensor, tf.Variable],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    return tf.random.shuffle(x)
