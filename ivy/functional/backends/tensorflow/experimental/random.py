# global
from typing import Union, Optional, Sequence
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.python.framework.dtypes import DType

# local
import ivy
from .. import backend_version
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _check_shapes_broadcastable,
)


# dirichlet
@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "blfoat16",
            "float16",
        )
    },
    backend_version,
)
def dirichlet(
    alpha: Union[tf.Tensor, tf.Variable, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    seed: Optional[int] = None,
    dtype: Optional[tf.Tensor] = None,
) -> Union[tf.Tensor, tf.Variable]:
    size = size if size is not None else len(alpha)

    if dtype is None:
        dtype = tf.float64
    else:
        dtype = dtype
    if seed is not None:
        tf.random.set_seed(seed)
    return tf.cast(
        tfd.Dirichlet(
            concentration=alpha,
            validate_args=False,
            allow_nan_stats=True,
            force_probs_to_zero_outside_support=False,
            name="Dirichlet",
        ).sample(size),
        dtype=dtype,
    )


def beta(
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


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, backend_version)
def poisson(
    lam: Union[float, tf.Tensor, tf.Variable],
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str,
    dtype: DType,
    seed: Optional[int] = None,
    fill_value: Optional[Union[float, int]] = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    lam = tf.cast(lam, "float32")
    if seed:
        tf.random.set_seed(seed)
    if shape is None:
        return tf.random.poisson((), lam, dtype=dtype, seed=seed)
    shape = tf.cast(shape, "int32")
    _check_shapes_broadcastable(lam.shape, shape)
    lam = tf.broadcast_to(lam, tuple(shape))
    ret = tf.random.poisson((), lam, dtype=dtype, seed=seed)
    if tf.reduce_any(lam < 0):
        return tf.where(lam < 0, fill_value, ret)
    return ret


def bernoulli(
    probs: Union[float, tf.Tensor, tf.Variable],
    *,
    logits: Union[float, tf.Tensor, tf.Variable] = None,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: str,
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
