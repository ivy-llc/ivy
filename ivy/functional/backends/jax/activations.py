"""Collection of Jax activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional

# global
import jax
import jax.numpy as jnp
import numpy as np

# local
from jax import lax

import ivy
from ivy.functional.backends.jax import JaxArray


def relu(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.maximum(x, 0)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def leaky_relu(x: JaxArray, alpha: Optional[float] = 0.2) -> JaxArray:
    return jnp.where(x > 0, x, x * alpha)


def gelu(x: JaxArray, approximate: Optional[bool] = False)->JaxArray:

    if approximate:
        sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
        cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
        return x * cdf
    else:
        return jnp.array(x * (lax.erf(x / np.sqrt(2)) + 1) / 2)

def sigmoid(x: JaxArray) -> JaxArray:
    return 1 / (1 + jnp.exp(-x))


def tanh(x: JaxArray) -> JaxArray:
    return jnp.tanh(x)


def softmax(x: JaxArray, axis: Optional[int] = None) -> JaxArray:
    exp_x = jnp.exp(x)
    return exp_x / jnp.sum(exp_x, axis, keepdims=True)


def softplus(x: JaxArray) -> JaxArray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)
