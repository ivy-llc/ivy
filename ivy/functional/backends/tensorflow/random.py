"""Collection of TensorFlow random functions, wrapped to fit Ivy syntax and
signature.
"""

# global
import tensorflow as tf
from typing import Optional, Union, Tuple, Sequence

# local
from ivy.functional.ivy.device import default_device


# Extra #
# ------#


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype=None,
    *,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    low = tf.cast(low, dtype)
    high = tf.cast(high, dtype)
    with tf.device(default_device(device)):
        return tf.random.uniform(shape if shape else (), low, high, dtype=dtype)


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    mean = tf.cast(mean, "float32")
    std = tf.cast(std, "float32")
    with tf.device(default_device(device)):
        return tf.random.normal(shape if shape else (), mean, std)


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[Union[tf.Tensor, tf.Variable]] = None,
    replace: bool = True,
    *,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    if not replace:
        raise Exception("TensorFlow does not support multinomial without replacement")
    device = default_device(device)
    with tf.device("/" + device.upper()):
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
    shape: Union[int, Sequence[int]], 
    *, 
    device: str
) -> Union[tf.Tensor, tf.Variable]:
    device = default_device(device)
    low = tf.cast(low, "int64")
    high = tf.cast(high, "int64")
    with tf.device("/" + device.upper()):
        return tf.random.uniform(shape=shape, minval=low, maxval=high, dtype=tf.int64)


def seed(seed_value: int = 0) -> None:
    tf.random.set_seed(seed_value)


def shuffle(x: Union[tf.Tensor, tf.Variable]) -> Union[tf.Tensor, tf.Variable]:
    return tf.random.shuffle(x)
