"""Collection of TensorFlow random functions, wrapped to fit Ivy syntax and
signature.
"""

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Optional, Union, Tuple

# local
import ivy
from ivy.functional.ivy.device import default_device


# Extra #
# ------#


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    device: Optional[ivy.Device] = None,
) -> Tensor:
    with tf.device(default_device(device)):
        return tf.random.uniform(shape if shape else (), low, high)


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    device: Optional[ivy.Device] = None,
) -> Tensor:
    with tf.device(default_device(device)):
        return tf.random.normal(shape if shape else (), mean, std)


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[Tensor] = None,
    replace: bool = True,
    device: Optional[ivy.Device] = None,
) -> Tensor:
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


def randint(low, high, shape, device=None):
    device = default_device(device)
    with tf.device("/" + device.upper()):
        return tf.random.uniform(shape=shape, minval=low, maxval=high, dtype=tf.int32)


seed = lambda seed_value=0: tf.random.set_seed(seed_value)


def shuffle(x: Tensor) -> Tensor:
    return tf.random.shuffle(x)
