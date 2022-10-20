import logging
from typing import Optional, Union, Tuple, Callable, Literal, Sequence
from numbers import Number
import ivy
from ivy.functional.ivy.extensions import (
    _verify_coo_components,
    _verify_csr_components,
    _is_coo_not_csr,
)
from ivy.functional.backends.jax import JaxArray
import jax.lax as jlax
import jax.numpy as jnp
import math


def lcm(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.lcm(x1, x2)


def sinc(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinc(x)


def fmod(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fmod(x1, x2)


def fmax(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fmax(x1, x2)
