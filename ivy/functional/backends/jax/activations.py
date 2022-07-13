"""Collection of Jax activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional

# global
import jax
import jax.numpy as jnp

# local
import ivy
from ivy.functional.backends.jax import JaxArray


def relu(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.maximum(x, 0)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def leaky_relu(
        x: JaxArray,
        alpha: Optional[float] = 0.2,
        out: Optional[JaxArray] = None
) -> JaxArray:
    ret = jnp.where(x > 0, x, x * alpha)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def gelu(
        x: JaxArray,
        approximate: Optional[bool] = True,
        out: Optional[JaxArray] = None
) -> JaxArray:
    ret = jax.nn.gelu(x, approximate)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sigmoid(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = 1 / (1 + jnp.exp(-x))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def tanh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.tanh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def softmax(
        x: JaxArray,
        axis: Optional[int] = None,
        out: Optional[JaxArray] = None
) -> JaxArray:
    exp_x = jnp.exp(x)
    ret = exp_x / jnp.sum(exp_x, axis, keepdims=True)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def softplus(x: JaxArray, out: Optional[JaxArray]= None) -> JaxArray:
    ret = jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret
