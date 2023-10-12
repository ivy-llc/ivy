# global
from typing import Union, Optional, Literal

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.utils.exceptions import handle_exceptions


def _logit_jax_like(
    x: Union[float, int, ivy.Array],
    /,
    *,
    eps: Optional[float] = None,
    out: Optional[ivy.Array] = None,
):
    real = ivy.real(x)
    imag = ivy.imag(x)
    if eps is None:
        real = ivy.where(ivy.logical_or(real > 1, real < 0), ivy.nan, real)
    else:
        real = ivy.clip(real, eps, 1 - eps)
    z = ivy.add(real, ivy.multiply(ivy.array(1j, dtype=x.dtype), imag))
    z = ivy.log(z / (1 - z))
    return z


def logit(
    x: Union[float, int, ivy.Array],
    /,
    *,
    eps: Optional[float] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return current_backend(x).logit(x, eps=eps, out=out)


logit.jax_like = _logit_jax_like


@handle_exceptions
def prelu(
    x: ivy.Array,
    slope: ivy.Array,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.where(x > 0, x, x * slope, out=out)


@handle_exceptions
def thresholded_relu(
    x: ivy.Array,
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.where(x > threshold, x, ivy.zeros_like(x), out=out)


@handle_exceptions
def relu6(
    x: ivy.Array,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.where(
        x < 0,
        ivy.zeros_like(x),
        ivy.where(x > 6, ivy.array(6, dtype=x.dtype), x),
        out=out,
    )


@handle_exceptions
def logsigmoid(
    x: ivy.Array,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.log(1 / (1 + ivy.exp(-x)), out=out)


@handle_exceptions
def selu(
    x: ivy.Array,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    scale = 1.0507
    return ivy.where(x > 0, x, scale * (ivy.exp(x) - 1), out=out)


@handle_exceptions
def silu(
    x: ivy.Array,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return x * ivy.sigmoid(x, out=out)


@handle_exceptions
def elu(
    x: ivy.Array,
    /,
    *,
    alpha: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.where(x > 0, x, alpha * (ivy.exp(x) - 1), out=out)
