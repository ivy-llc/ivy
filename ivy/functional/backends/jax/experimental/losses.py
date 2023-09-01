import jax.numpy as jnp
from typing import Optional
from ivy.functional.backends.jax import JaxArray


def huber_loss(
    input: JaxArray, target: JaxArray, /, *, delta: float = 1.0, reduction: str = "mean"
) -> JaxArray:
    residual = jnp.abs(input - target)
    quadratic_loss = 0.5 * (residual**2)
    linear_loss = delta * residual - 0.5 * (delta**2)
    loss = jnp.where(residual < delta, quadratic_loss, linear_loss)

    if reduction == "mean":
        loss = jnp.mean(loss)
    elif reduction == "sum":
        loss = jnp.sum(loss)

    return loss


def smooth_l1_loss(
    input: JaxArray,
    target: JaxArray,
    /,
    *,
    beta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
) -> JaxArray:
    if beta < 1e-5:
        loss = jnp.abs(input - target)
    else:
        diff = jnp.abs(input - target)
        loss = jnp.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        return loss


def soft_margin_loss(
    input: JaxArray,
    target: JaxArray,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> JaxArray:
    loss = jnp.sum(jnp.log1p(jnp.exp(-input * target))) / jnp.size(input)

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        return loss
