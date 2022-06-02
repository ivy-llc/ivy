"""Collection of Jax activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional

# global
import jax
import jax.numpy as jnp

# local
import ivy
from ivy.functional.backends.jax import JaxArray


def relu(x: JaxArray) -> JaxArray:
    return jnp.maximum(x, 0)


def leaky_relu(x: JaxArray, alpha: Optional[float] = 0.2) -> JaxArray:
    return jnp.where(x > 0, x, x * alpha)


gelu = jax.nn.gelu


def sigmoid(x: JaxArray) -> JaxArray:
    return 1 / (1 + jnp.exp(-x))


def tanh(x: JaxArray) -> JaxArray:
    return jnp.tanh


def softmax(x: JaxArray, axis: Optional[int] = -1) -> JaxArray:
    exp_x = jnp.exp(x)
    return exp_x / jnp.sum(exp_x, axis, keepdims=True)


def softplus(x: JaxArray) -> JaxArray:
    return jnp.log(jnp.exp(x) + 1)
