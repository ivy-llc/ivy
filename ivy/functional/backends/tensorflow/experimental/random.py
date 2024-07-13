# global
from typing import Union, Optional, Sequence
import tensorflow as tf
from tensorflow.python.framework.dtypes import DType

# local
import ivy
from .. import backend_version
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.ivy.random import (
    _check_shapes_broadcastable,
)


# dirichlet
@with_unsupported_dtypes(
    {
        "2.15.0 and below": (
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
    pass
    # TODO: Implement purely in tensorflow


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
    pass
    # TODO: Implement purely in tensorflow


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
    pass
    # TODO: Implement purely in tensorflow


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16",)}, backend_version)
def poisson(
    lam: Union[float, tf.Tensor, tf.Variable],
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Optional[str] = None,
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


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16",)}, backend_version)
def bernoulli(
    probs: Union[float, tf.Tensor, tf.Variable],
    *,
    logits: Union[float, tf.Tensor, tf.Variable] = None,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    seed: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = dtype if dtype is not None else probs.dtype
    if logits is not None:
        probs = tf.nn.softmax(logits, -1)
    if not _check_shapes_broadcastable(shape, probs.shape):
        shape = probs.shape
    return tf.keras.backend.random_bernoulli(shape, probs, dtype, seed)
