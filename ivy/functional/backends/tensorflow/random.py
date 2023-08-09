"""
TensorFlow random functions.

Collection of TensorFlow random functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional, Union, Sequence

# global
import tensorflow as tf
from tensorflow.python.framework.dtypes import DType

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)
from . import backend_version


# Extra #
# ------#


def random_uniform(
    *,
    low: Union[float, tf.Tensor, tf.Variable] = 0.0,
    high: Union[float, tf.Tensor, tf.Variable] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int], tf.Tensor]] = None,
    dtype: DType,
    device: str,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    shape = _check_bounds_and_get_shape(low, high, shape).shape
    low = tf.cast(low, dtype)
    high = tf.cast(high, dtype)
    if seed:
        tf.random.set_seed(seed)
    return tf.random.uniform(shape, low, high, dtype=dtype, seed=seed)


def random_normal(
    *,
    mean: Union[float, tf.Tensor, tf.Variable] = 0.0,
    std: Union[float, tf.Tensor, tf.Variable] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: DType,
    seed: Optional[int] = None,
    device: str,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape).shape
    mean = tf.cast(mean, dtype)
    std = tf.cast(std, dtype)
    if seed:
        tf.random.set_seed(seed)
    return tf.random.normal(shape, mean, std, dtype=dtype, seed=seed)


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, backend_version)
def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[Union[tf.Tensor, tf.Variable]] = None,
    replace: bool = True,
    device: str,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
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

    # We set the global seed, but not the operation seeds below. In this way, we
    # get different results for every random op call but the same sequence for
    # every re-run of the program
    if seed:
        tf.random.set_seed(seed)

    if not replace:
        orig_probs_shape = list(probs.shape)
        probs_flat = tf.reshape(probs, (-1, orig_probs_shape[-1]))
        probs_flat = probs_flat / tf.math.reduce_sum(
            probs_flat, axis=-1, keepdims=True
        )
        probs_stack = tf.split(probs_flat, probs_flat.shape[0])
        samples_stack = []
        for prob in probs_stack:
            logits = tf.dtypes.cast(tf.math.log(prob), tf.float64)
            # Gumbel-max trick
            # https://github.com/tensorflow/tensorflow/issues/9260
            z = tf.dtypes.cast(
                -tf.math.log(
                    -tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1))
                ),
                tf.float64,
            )
            _, indices = tf.nn.top_k(logits + z, k=num_samples)
            samples_stack.append(indices)
        samples_flat = tf.stack(samples_stack)
        return tf.convert_to_tensor(
            tf.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples])
        )
    else:
        if len(probs.numpy().shape) == 1:
            probs = tf.expand_dims(probs, axis=0)
        return tf.random.categorical(tf.math.log(probs), num_samples)


def randint(
    low: Union[float, tf.Tensor, tf.Variable],
    high: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str,
    dtype: Optional[Union[DType, ivy.Dtype]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape).shape
    low = tf.cast(low, "float32")
    high = tf.cast(high, "float32")
    if seed:
        tf.random.set_seed(seed)
    return tf.cast(tf.random.uniform(shape, low, high, "float32", seed=seed), dtype)


def seed(*, seed_value: int = 0) -> None:
    tf.random.set_seed(seed_value)
    return


def shuffle(
    x: Union[tf.Tensor, tf.Variable],
    axis: Optional[int] = 0,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if seed:
        tf.random.set_seed(seed)
    return tf.random.shuffle(x, seed=seed)
