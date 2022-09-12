"""Collection of Jax activation functions, wrapped to fit Ivy syntax and signature."""
# global
import inspect
from typing import Optional

import jax
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray
from . import dtype_from_version, jax_version


class VersionedAttributes:
    def __init__(self, attribute_function):
        self.attribute_function = attribute_function

    def __get__(self, obj, objtype=None):
        return self.attribute_function()

    def __iter__(self):
        return iter(self.attribute_function())


# Decorator to set unsupported dtypes
def _with_unsupported_dtypes(version_dict, version):
    def _unsupported_wrapper(func):
        func.unsupported_dtypes = VersionedAttributes(lambda: dtype_from_version(version_dict, version["version"]))
        return func

    return _unsupported_wrapper


@_with_unsupported_dtypes({"0.3.14": ("uint32",), "0.1": ("float64",)}, jax_version)
def relu(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.maximum(x, 0)




def leaky_relu(
    x: JaxArray, /, *, alpha: Optional[float] = 0.2, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.where(x > 0, x, x * alpha)


def gelu(
    x: JaxArray,
    /,
    *,
    approximate: Optional[bool] = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jax.nn.gelu(x, approximate)


def sigmoid(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return 1 / (1 + jnp.exp(-x))


def softmax(
    x: JaxArray, /, *, axis: Optional[int] = None, out: Optional[JaxArray] = None
) -> JaxArray:
    if axis is None:
        axis = -1
    return jax.nn.softmax(x, axis)


def softplus(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)
