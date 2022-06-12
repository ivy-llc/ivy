# global
import jax
import jax.numpy as jnp
from typing import Optional

# local
from ivy.functional.backends.jax import JaxArray


def bitwise_left_shift(x1: JaxArray, x2: JaxArray) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.left_shift(x1, x2)


def add(x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.add(x1, x2)


def bitwise_xor(x1: JaxArray, x2: JaxArray) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.bitwise_xor(x1, x2)


def exp(x: JaxArray) -> JaxArray:
    return jnp.exp(x)


def expm1(x: JaxArray) -> JaxArray:
    return jnp.expm1(x)


def bitwise_invert(x: JaxArray) -> JaxArray:
    return jnp.bitwise_not(x)


def bitwise_and(x1: JaxArray, x2: JaxArray) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.bitwise_and(x1, x2)


def ceil(x: JaxArray) -> JaxArray:
    if "int" in str(x.dtype):
        ret = x
    else:
        ret = jnp.ceil(x)
    return ret


def floor(x: JaxArray
         ) -> JaxArray:
    if "int" in str(x.dtype):
        ret = x
    else:
        ret = jnp.floor(x)
    return ret


def isfinite(x: JaxArray) -> JaxArray:
    return jnp.isfinite(x)


def asin(x: JaxArray) -> JaxArray:
    return jnp.arcsin(x)


def isinf(x: JaxArray) -> JaxArray:
    return jnp.isinf(x)


def equal(x1: JaxArray, x2: JaxArray) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.equal(x1, x2)


def greater(x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.greater(x1, x2)


def greater_equal(x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.greater_equal(x1, x2)


def less_equal(x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.less_equal(x1, x2)


def asinh(x: JaxArray) -> JaxArray:
    return jnp.arcsinh(x)


def sign(x: JaxArray) -> JaxArray:
    return jnp.sign(x)


def sqrt(x: JaxArray) -> JaxArray:
    return jnp.sqrt(x)


def cosh(x: JaxArray) -> JaxArray:
    return jnp.cosh(x)


def log10(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log10(x)


def log(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log(x)


def log2(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log2(x)


def log1p(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log1p(x)


def multiply(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.multiply(x1, x2)


def isnan(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isnan(x)


def less(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.less(x1, x2)


def cos(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.cos(x)


def logical_xor(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.logical_xor(x1, x2)


def logical_or(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.logical_or(x1, x2)


def logical_and(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.logical_and(x1, x2)


def logical_not(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.logical_not(x)


def divide(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.divide(x1, x2)


def acos(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arccos(x)


def acosh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arccosh(x)


def sin(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sin(x)


def negative(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.negative(x)


def not_equal(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.not_equal(x1, x2)


def tanh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.tanh(x)


def floor_divide(
    x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None
) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.floor_divide(x1, x2)


def bitwise_or(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.bitwise_or(x1, x2)


def sinh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinh(x)


def positive(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.positive(x)


def square(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.square(x)


def pow(
    x1: jnp.ndarray, x2: jnp.ndarray, out: Optional[JaxArray] = None
) -> jnp.ndarray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    return jnp.power(x1, x2)


def remainder(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.remainder(x1, x2)


def round(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.round(x)


def trunc(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.trunc(x)


def abs(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.absolute(x)


def subtract(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    return jnp.subtract(x1, x2)


def logaddexp(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.logaddexp(x1, x2)


def bitwise_right_shift(
    x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None
) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.right_shift(x1, x2)


def tan(x: JaxArray) -> JaxArray:
    return jnp.tan(x)


def atan(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arctan(x)


def atanh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arctanh(x)


def atan2(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    return jnp.arctan2(x1, x2)


# Extra #
# ------#


def minimum(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.minimum(x1, x2)


def maximum(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.maximum(x1, x2)


def erf(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.scipy.special.erf(x)
