from typing import Optional, Union

# global
import jax
import jax.numpy as jnp
from ivy.functional.backends.jax import JaxArray
from jax import lax
import ivy


def logit(
    x: JaxArray,
    /,
    *,
    eps: Optional[float] = None,
    out: Optional[JaxArray] = None,
):
    if eps is None:
        x = jnp.where(jnp.logical_or(x > 1, x < 0), jnp.nan, x)
    else:
        x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x / (1 - x))


def relu6(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    relu6_func = jax.nn.relu6

    # sets gradient at 0 and 6 to 0 instead of 0.5
    # can refactor to jax.nn.relu6 when this PR is merged
    # https://github.com/google/jax/pull/14682
    def custom_grad_func(x_and_grad, one):
        return lax.select(
            (6 > x_and_grad[0]) & (x_and_grad[0] > 0), one, lax.full_like(one, 0)
        )

    new_func = ivy.bind_custom_gradient_function(relu6_func, custom_grad_func)

    return new_func(x).astype(x.dtype)


def thresholded_relu(
    x: JaxArray,
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x, threshold = ivy.promote_types_of_inputs(x, threshold)
    return jnp.where(x > threshold, x, 0).astype(x.dtype)


def batch_norm(
    x: JaxArray,
    mean: JaxArray,
    variance: JaxArray,
    /,
    *,
    scale: Optional[JaxArray] = None,
    offset: Optional[JaxArray] = None,
    training: bool = False,
    eps: float = 1e-5,
):
    ndims = len(x.shape)
    if training:
        dims = (0, *range(2, ndims))
        mean = jnp.mean(x, axis=dims)
        variance = jnp.var(x, axis=dims)
    x = jnp.transpose(x, (0, *range(2, ndims), 1))
    inv = 1.0 / jnp.sqrt(variance + eps)
    if scale is not None:
        inv *= scale

    ret = x * inv.astype(x.dtype) + (
        offset - mean * inv if offset is not None else -mean * inv
    ).astype(x.dtype)

    return jnp.transpose(ret, (0, ndims - 1, *range(1, ndims - 1)))


def logsigmoid(input: JaxArray) -> JaxArray:
    return jax.nn.log_sigmoid(input)


def selu(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jax.nn.selu(x).astype(x.dtype)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret).astype(x.dtype)
    return ret
