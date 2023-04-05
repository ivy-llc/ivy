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


def hardshrink(
    x: JaxArray,
    /,
    *,
    lambd: Optional[float] = 0.5,
    out: Optional[JaxArray] = None,
):
    mask = jnp.logical_or(jnp.greater(x, lambd), jnp.less(x, -lambd))
    return jnp.where(mask, x, 0.0)


def softshrink(
    x: JaxArray,
    /,
    *,
    lambd: Optional[float] = 0.5,
    out: Optional[JaxArray] = None,
):
    low = jnp.where(jnp.less(x, -lambd), jnp.add(x, lambd), 0)
    up = jnp.where(jnp.greater(x, lambd), jnp.subtract(x, lambd), 0)
    return jnp.add(low, up)


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
    return jnp.where(x > threshold, x, 0).astype(x.dtype)


def threshold(
    x: JaxArray,
    threshold: Union[int, float],
    value: Union[int, float],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.where(x > threshold, x, value).astype(x.dtype)


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


def logsigmoid(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.nn.log_sigmoid(x)


def hard_tanh(
    x: JaxArray,
    /,
    *,
    min_value: float = -1.0,
    max_value: float = 1.0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.clip(x, a_min=min_value, a_max=max_value)


def softsign(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.nn.soft_sign(x)


def silu(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.nn.silu(x)


def selu(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.nn.selu(x)


def elu(
    x: JaxArray, /, *, alpha: float = 1.0, out: Optional[JaxArray] = None
) -> JaxArray:
    return jax.nn.elu(x, alpha=alpha)


def parametric_relu(
    x: JaxArray, weight: Union[float, JaxArray], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.where(x >= 0, x, weight * x).astype(x.dtype)


def celu(
    x: JaxArray, /, *, alpha: float = 1.0, out: Optional[JaxArray] = None
) -> JaxArray:
    return jax.nn.celu(x, alpha=alpha)


def hard_sigmoid(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.nn.hard_sigmoid(x)


def hard_silu(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.nn.hard_silu(x)


def glu(x: JaxArray, /, *, axis: int = -1, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.nn.glu(x, axis=axis)
