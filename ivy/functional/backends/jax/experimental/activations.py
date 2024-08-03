from typing import Optional, Union, Literal

# global
import jax
import jax.numpy as jnp
from ivy.functional.backends.jax import JaxArray
from jax import lax
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def logit(
    x: JaxArray,
    /,
    *,
    eps: Optional[float] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[JaxArray] = None,
):
    if eps is None:
        x = jnp.where(jnp.logical_or(x > 1, x < 0), jnp.nan, x)
    else:
        x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x / (1 - x))


def relu6(
    x: JaxArray, /, *, complex_mode="jax", out: Optional[JaxArray] = None
) -> JaxArray:
    relu6_func = jax.nn.relu6

    # sets gradient at 0 and 6 to 0 instead of 0.5
    # can refactor to jax.nn.relu6 when this PR is merged
    # https://github.com/google/jax/pull/14682
    def custom_grad_func(x_and_grad, one):
        return lax.select(
            (x_and_grad[0] < 6) & (x_and_grad[0] > 0), one, lax.full_like(one, 0)
        )

    new_func = ivy.bind_custom_gradient_function(relu6_func, custom_grad_func)

    return jnp.astype(new_func(x), x.dtype)


def thresholded_relu(
    x: JaxArray,
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.astype(jnp.where(x > threshold, x, 0), x.dtype)


def logsigmoid(
    input: JaxArray, /, *, complex_mode="jax", out: Optional[JaxArray] = None
) -> JaxArray:
    return jax.nn.log_sigmoid(input)


def selu(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.astype(jax.nn.selu(x), x.dtype)
    if ivy.exists(out):
        return jnp.astype(ivy.inplace_update(out, ret), x.dtype)
    return ret


def silu(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jax.nn.silu(x)
    if ivy.exists(out):
        return jnp.astype(ivy.inplace_update(out, ret), x.dtype)
    return ret


@with_unsupported_dtypes({"0.4.14 and below": ("float16", "bfloat16")}, backend_version)
def elu(
    x: JaxArray, /, *, alpha: float = 1.0, out: Optional[JaxArray] = None
) -> JaxArray:
    ret = jax.nn.elu(x, alpha)
    if ivy.exists(out):
        return jnp.astype(ivy.inplace_update(out, ret), x.dtype)
    return ret


def celu(
    x: JaxArray,
    /,
    *,
    alpha: float = 1.0,
    complex_mode="jax",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jax.nn.celu(x, alpha=alpha)


@with_unsupported_dtypes({"0.4.14 and below": ("float16", "bfloat16")}, backend_version)
def hardtanh(
    x: JaxArray,
    /,
    *,
    max_val: float = 1.0,
    min_val: float = -1.0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ret = jnp.where(x > max_val, max_val, jnp.where(x < min_val, min_val, x))
    if ivy.exists(out):
        return ivy.astype(ivy.inplace_update(out, ret), x.dtype)
    return ivy.astype(ret, x.dtype)


def tanhshrink(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.subtract(x, jax.nn.tanh(x))
    if ivy.exists(out):
        return jnp.astype(ivy.inplace_update(out, ret), x.dtype)
    return ret


def threshold(
    x: JaxArray,
    /,
    *,
    threshold: Union[int, float],
    value: Union[int, float],
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ret = jnp.astype(jnp.where(x > threshold, x, value), x.dtype)
    if ivy.exists(out):
        return jnp.astype(ivy.inplace_update(out, ret), x.dtype)  # type: ignore
    return ret


@with_unsupported_dtypes({"0.4.16 and below": ("float16", "bfloat16")}, backend_version)
def softshrink(
    x: JaxArray, /, *, lambd: float = 0.5, out: Optional[JaxArray] = None
) -> JaxArray:
    ret = jnp.where(x > lambd, x - lambd, jnp.where(x < -lambd, x + lambd, 0))
    if ivy.exists(out):
        return jnp.astype(ivy.inplace_update(out, ret), x.dtype)
    return ret


@with_unsupported_dtypes({"0.4.17 and below": ("float64",)}, backend_version)
def scaled_tanh(
    x: JaxArray,
    /,
    *,
    alpha: float = 1.7159,
    beta: float = 0.67,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return alpha * jax.nn.tanh(beta * x)


@with_unsupported_dtypes({"0.4.16 and below": ("float16", "bfloat16")}, backend_version)
def hardshrink(
    x: JaxArray, /, *, lambd: float = 0.5, out: Optional[JaxArray] = None
) -> JaxArray:
    ret = jnp.where(x > lambd, x, jnp.where(x < -lambd, x, 0))
    if ivy.exists(out):
        return jnp.astype(ivy.inplace_update(out, ret), x.dtype)
    return ret


@with_unsupported_dtypes({"0.4.16 and below": ("float16", "bfloat16")}, backend_version)
def hardsilu(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jax.nn.hard_silu(x)
    if ivy.exists(out):
        return jnp.astype(ivy.inplace_update(out, ret), x.dtype)
    return ret
