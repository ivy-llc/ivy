from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _check_shapes_broadcastable,
)
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.python.framework.dtypes import DType

from typing import Optional, Sequence, Union

import ivy


def beta(
    alpha: Union[float, tf.Tensor, tf.Variable],
    beta: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Optional[str] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.Dtype]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not dtype:
        dtype = ivy.default_float_dtype()
    dtype = ivy.as_native_dtype(dtype)
    shape = _check_bounds_and_get_shape(alpha, beta, shape).shape
    alpha = tf.cast(alpha, dtype)
    beta = tf.cast(beta, dtype)
    return tfp.distributions.Beta(alpha, beta).sample(shape, seed=seed)


def gamma(
    alpha: Union[float, tf.Tensor, tf.Variable],
    beta: Union[float, tf.Tensor, tf.Variable],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Optional[str] = None,
    dtype: Optional[Union[DType, ivy.Dtype]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not dtype:
        dtype = ivy.default_float_dtype()
    dtype = ivy.as_native_dtype(dtype)
    shape = _check_bounds_and_get_shape(alpha, beta, shape).shape
    alpha = tf.cast(alpha, dtype)
    beta = tf.cast(beta, dtype)
    return tfp.distributions.Gamma(alpha, beta).sample(shape, seed=seed)


def bernoulli(
    probs: Union[float, tf.Tensor, tf.Variable],
    *,
    logits: Union[float, tf.Tensor, tf.Variable] = None,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str = None,
    dtype: DType,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if seed is not None:
        tf.random.set_seed(seed)
    if logits is not None:
        logits = tf.cast(logits, dtype)
        if not _check_shapes_broadcastable(shape, logits.shape):
            shape = logits.shape
    elif probs is not None:
        probs = tf.cast(probs, dtype)
        if not _check_shapes_broadcastable(shape, probs.shape):
            shape = probs.shape
    return tfp.distributions.Bernoulli(
        logits=logits, probs=probs, dtype=dtype, allow_nan_stats=True
    ).sample(shape, seed)
