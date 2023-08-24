
import jax.numpy as jnp
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
