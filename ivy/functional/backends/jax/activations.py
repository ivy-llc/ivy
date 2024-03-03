"""Collection of Jax activation functions, wrapped to fit Ivy syntax and
signature."""

# global


import jax
import jax.numpy as jnp
from typing import Optional, Union, Literal

# local
from ivy.functional.backends.jax import JaxArray


def gelu(
    x: JaxArray,
    /,
    *,
    approximate: bool = False,
    complex_mode="jax",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jax.nn.gelu(x, approximate)


def leaky_relu(
    x: JaxArray,
    /,
    *,
    alpha: float = 0.2,
    complex_mode="jax",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.asarray(jnp.where(x > 0, x, jnp.multiply(x, alpha)), x.dtype)


def relu(
    x: JaxArray, /, *, complex_mode="jax", out: Optional[JaxArray] = None
) -> JaxArray:
    return jax.nn.relu(x)


def sigmoid(
    x: JaxArray, /, *, complex_mode="jax", out: Optional[JaxArray] = None
) -> JaxArray:
    return 1 / (1 + jnp.exp(-x))


def softmax(
    x: JaxArray, /, *, axis: Optional[int] = None, out: Optional[JaxArray] = None
) -> JaxArray:
    if axis is None:
        axis = -1
    return jax.nn.softmax(x, axis)


def softplus(
    x: JaxArray,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    complex_mode="jax",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if beta is not None and beta != 1:
        x_beta = x * beta
        res = (
            jnp.add(
                jnp.log1p(jnp.exp(-jnp.abs(x_beta))),
                jnp.maximum(x_beta, 0).astype(x.dtype),
            )
        ) / beta
    else:
        x_beta = x
        res = jnp.add(
            jnp.log1p(jnp.exp(-jnp.abs(x_beta))),
            jnp.maximum(x_beta, 0).astype(x.dtype),
        )
    if threshold is not None:
        return jnp.where(x_beta > threshold, x, res).astype(x.dtype)
    return res.astype(x.dtype)


# Softsign
def softsign(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.nn.soft_sign(x)


def log_softmax(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = -1,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[JaxArray] = None,
):
    return jax.nn.log_softmax(x, axis)


def mish(
    x: JaxArray,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return x * jnp.tanh(jax.nn.softplus(x))


def hardswish(
    x: JaxArray,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jax.nn.hard_swish(x)
