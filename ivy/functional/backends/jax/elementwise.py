# global
from typing import Union, Optional

import jax
import jax.numpy as jnp

# local
import ivy
from ivy.functional.backends.jax import JaxArray


def _cast_for_bitwise_op(x1, x2, clamp=False):
    if not isinstance(x1, int):
        if isinstance(x2, int):
            x2 = jnp.asarray(x2, dtype=x1.dtype)
    if clamp:
        x2 = jax.lax.clamp(
            jnp.array(0, dtype=x2.dtype),
            x2,
            jnp.array(x1.dtype.itemsize * 8 - 1, dtype=x2.dtype),
        )
    return x1, x2


def _cast_for_binary_op(x1, x2):
    return ivy.promote_types_of_inputs(x1, x2)


def abs(x: Union[float, JaxArray], *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.absolute(x)


def acos(x: JaxArray) -> JaxArray:
    return jnp.arccos(x)


def acosh(x: JaxArray) -> JaxArray:
    return jnp.arccosh(x)


def add(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.add(x1, x2)


def asin(x: JaxArray, *, out: Union[float, JaxArray] = None) -> JaxArray:
    return jnp.arcsin(x)


def asinh(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arcsinh(x)


def atan(x: JaxArray) -> JaxArray:
    return jnp.arctan(x)


def atan2(x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jnp.arctan2(x1, x2)


def atanh(x: JaxArray) -> JaxArray:
    return jnp.arctanh(x)


def bitwise_and(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_bitwise_op(x1, x2)
    return jnp.bitwise_and(x1, x2)


def bitwise_invert(
    x: Union[int, JaxArray], *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.bitwise_not(x)


def bitwise_left_shift(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_bitwise_op(x1, x2, clamp=True)
    return jnp.left_shift(x1, x2)


def bitwise_or(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_bitwise_op(x1, x2)
    return jnp.bitwise_or(x1, x2)


def bitwise_right_shift(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_bitwise_op(x1, x2, clamp=True)
    return jnp.right_shift(x1, x2)


def bitwise_xor(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_bitwise_op(x1, x2)
    return jnp.bitwise_xor(x1, x2)


def ceil(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.ceil(x)


def cos(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.cos(x)


def cosh(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.cosh(x)


def divide(x1: Union[float, JaxArray], x2: Union[float, JaxArray]) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    ret = jax.numpy.divide(x1, x2)
    if ivy.is_float_dtype(x1.dtype):
        ret = jnp.asarray(ret, dtype=x1.dtype)
    else:
        ret = jnp.asarray(ret, dtype=ivy.default_float_dtype(as_native=True))
    return ret


def equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jnp.equal(x1, x2)


def exp(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.exp(x)


def expm1(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.expm1(x)


def floor(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.floor(x)


def floor_divide(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jax.numpy.floor_divide(x1, x2)


def greater(x1: Union[float, JaxArray], x2: Union[float, JaxArray]) -> JaxArray:
    return jnp.greater(x1, x2)


def greater_equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.greater_equal(x1, x2)


def isfinite(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isfinite(x)


def isinf(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isinf(x)


def isnan(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isnan(x)


def less(x1: Union[float, JaxArray], x2: Union[float, JaxArray]) -> JaxArray:
    return jnp.less(x1, x2)


def less_equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.less_equal(x1, x2)


def log(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log(x)


def log10(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log10(x)


def log1p(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log1p(x)


def log2(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log2(x)


def logaddexp(
    x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logaddexp(x1, x2)


def logical_and(
    x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logical_and(x1, x2)


def logical_not(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.logical_not(x)


def logical_or(
    x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logical_or(x1, x2)


def logical_xor(
    x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logical_xor(x1, x2)


def multiply(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jnp.multiply(x1, x2)


def negative(x: Union[float, JaxArray], *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.negative(x)


def not_equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jnp.not_equal(x1, x2)


def positive(x: Union[float, JaxArray], *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.positive(x)


def pow(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jnp.power(x1, x2)


def remainder(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jnp.remainder(x1, x2)


def round(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.round(x)


def sign(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sign(x)


def sin(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sin(x)


def sinh(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinh(x)


def sqrt(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sqrt(x)


def square(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.square(x)


def subtract(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jnp.subtract(x1, x2)


def tan(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.tan(x)


def tanh(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.tanh(x)


def trunc(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.trunc(x)


# Extra #
# ------#


def erf(x: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.scipy.special.erf(x)


def maximum(x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jnp.maximum(x1, x2)


def minimum(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return jnp.minimum(x1, x2)
