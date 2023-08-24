import jax.numpy as jnp
from typing import Optional
from ivy.functional.backends.jax import JaxArray


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
