# global
from typing import Union, Optional

import jax
import jax.numpy as jnp

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.jax import JaxArray
from . import backend_version


def abs(x: Union[float, JaxArray], /, *, out: Optional[JaxArray] = None) -> JaxArray:
    # jnp.where is used for consistent gradients
    return jnp.where(x != 0, jnp.absolute(x), 0)


def acos(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arccos(x)


def acosh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arccosh(x)


def add(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    alpha: Optional[Union[int, float]] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        x2 = multiply(x2, alpha)
    return jnp.add(x1, x2)


def asin(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arcsin(x)


def asinh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arcsinh(x)


def atan(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arctan(x)


def atan2(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.arctan2(x1, x2)


def atanh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arctanh(x)


def bitwise_and(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.bitwise_and(x1, x2)


def bitwise_invert(
    x: Union[int, JaxArray], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.bitwise_not(x)


def bitwise_left_shift(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.left_shift(x1, x2)


def bitwise_or(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.bitwise_or(x1, x2)


def bitwise_right_shift(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.right_shift(x1, x2)


def bitwise_xor(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.bitwise_xor(x1, x2)


def ceil(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.ceil(x)


def cos(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.cos(x)


@with_unsupported_dtypes({"0.3.14 and below": ("float16",)}, backend_version)
def cosh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.cosh(x)


def divide(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = jax.numpy.divide(x1, x2)
    if ivy.is_float_dtype(x1.dtype):
        ret = jnp.asarray(ret, dtype=x1.dtype)
    else:
        ret = jnp.asarray(ret, dtype=ivy.default_float_dtype(as_native=True))
    return ret


def equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.equal(x1, x2)


def exp(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.exp(x)


def expm1(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.expm1(x)


def floor(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.floor(x)


def floor_divide(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.floor(jnp.divide(x1, x2)).astype(x1.dtype)


def greater(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.greater(x1, x2)


def greater_equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.greater_equal(x1, x2)


def isfinite(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isfinite(x)


def isinf(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isinf(x)


def isnan(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isnan(x)


def less(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.less(x1, x2)


def less_equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.less_equal(x1, x2)


def log(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log(x)


def log10(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log10(x)


def log1p(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log1p(x)


def log2(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log2(x)


def logaddexp(
    x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logaddexp(x1, x2)


def logical_and(
    x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logical_and(x1, x2)


def logical_not(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.logical_not(x)


def logical_or(
    x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logical_or(x1, x2)


def logical_xor(
    x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logical_xor(x1, x2)


def multiply(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.multiply(x1, x2)


def negative(
    x: Union[float, JaxArray], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.negative(x)


def not_equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.not_equal(x1, x2)


def positive(
    x: Union[float, JaxArray], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.positive(x)


def pow(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.power(x1, x2)


def remainder(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    modulus: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = jnp.where(res >= 0, jnp.floor(res), jnp.ceil(res))
        diff = res - res_floored
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        return jnp.round(diff * x2).astype(x1.dtype)
    return jnp.remainder(x1, x2)


def round(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.round(x)


def sign(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.where(x == -0.0, 0.0, jnp.sign(x)).astype(x.dtype)


def sin(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sin(x)


def sinh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinh(x)


def sqrt(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sqrt(x)


def square(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.square(x)


def subtract(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        x2 = multiply(x2, alpha)
    return jnp.subtract(x1, x2)


def tan(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.tan(x)


def tanh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.tanh(x)


def trunc(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.trunc(x)


# Extra #
# ------#


def erf(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.scipy.special.erf(x)


def maximum(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    use_where: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        return jnp.where(x1 >= x2, x1, x2)
    return jnp.maximum(x1, x2)


def minimum(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    use_where: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        return jnp.where(x1 <= x2, x1, x2)
    return jnp.minimum(x1, x2)


def reciprocal(
    x: Union[float, JaxArray], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.reciprocal(x)


def deg2rad(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.deg2rad(x)


def rad2deg(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.rad2deg(x)


def isreal(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isreal(x)
