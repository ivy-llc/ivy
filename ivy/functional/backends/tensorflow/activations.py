"""Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional, Union

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def gelu(
    x: Tensor, /, *, approximate: bool = False, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.gelu(x, approximate)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def leaky_relu(
    x: Tensor, /, *, alpha: float = 0.2, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.leaky_relu(x, alpha)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def relu(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.relu(x)


def sigmoid(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    if not ivy.is_array(x):
        x = float(x)
    return tf.nn.sigmoid(x)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def softmax(
    x: Tensor, /, *, axis: Optional[int] = None, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.softmax(x, axis)


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def softplus(
    x: Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    if beta is not None and beta != 1:
        x_beta = x * beta
        res = (tf.nn.softplus(x_beta)) / beta
    else:
        x_beta = x
        res = tf.nn.softplus(x)
    if threshold is not None:
        return tf.where(x_beta > threshold, x, res)
    return res


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def log_softmax(
    x: Tensor, /, *, axis: Optional[int] = None, out: Optional[Tensor] = None
):
    return tf.nn.log_softmax(x, axis)


def deserialize(
    name: Union[str, None], /, *, custom_objects: Optional[ivy.Dict] = None
) -> Union[ivy.Callable, None]:
    return tf.keras.activations.deserialize(name, custom_objects)


def get(
    identifier: Union[str, ivy.Callable, None],
    /,
    *,
    custom_objects: Optional[ivy.Dict] = None,
) -> Union[ivy.Callable, None]:
    if identifier is None:
        return tf.keras.activations.linear

    if isinstance(identifier, str):
        identifier = str(identifier)
        return ivy.deserialize(identifier, custom_objects=custom_objects)

    elif callable(identifier):
        return identifier
    else:
        raise TypeError(
            f"Could not interpret activation function identifier: {identifier}"
        )


@with_unsupported_dtypes({"2.9.1 and below": ("complex",)}, backend_version)
def mish(
    x: Tensor,
    /,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    return x * tf.math.tanh(tf.math.softplus(x))
