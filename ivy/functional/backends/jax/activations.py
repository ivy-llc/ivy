"""Collection of Jax activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional

# global
import jax
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def relu(
    x: JaxArray, 
    *, 
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.maximum(x, 0)


def leaky_relu(
    x: JaxArray, 
    alpha: Optional[float] = 0.2,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.where(x > 0, x, x * alpha)


def gelu(
    x: JaxArray, 
    approximate: Optional[bool] = True,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jax.nn.gelu(x, approximate)


def sigmoid(
    x: JaxArray,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return 1 / (1 + jnp.exp(-x))


def tanh(
    x: JaxArray,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.tanh(x)


def softmax(
    x: JaxArray, 
    axis: Optional[int] = None,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    exp_x = jnp.exp(x)
    return exp_x / jnp.sum(exp_x, axis, keepdims=True)


def softplus(
    x: JaxArray,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.log(jnp.exp(x) + 1)
