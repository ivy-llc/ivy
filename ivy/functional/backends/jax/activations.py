# for review
"""Collection of Jax activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional

# global
import jax
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def relu(x: JaxArray, /) -> JaxArray:
    return jnp.maximum(x, 0)


def leaky_relu(x: JaxArray, /, *, alpha: Optional[float] = 0.2) -> JaxArray:
    return jnp.where(x > 0, x, x * alpha)


def gelu(
    x: JaxArray,
    /,
    *,
    approximate: Optional[bool] = True,
) -> JaxArray:
    return jax.nn.gelu(x, approximate)


def sigmoid(x: JaxArray, /) -> JaxArray:
    return 1 / (1 + jnp.exp(-x))


def softmax(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
) -> JaxArray:
    exp_x = jnp.exp(x)
    return exp_x / jnp.sum(exp_x, axis, keepdims=True)


def softplus(x: JaxArray, /) -> JaxArray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)
