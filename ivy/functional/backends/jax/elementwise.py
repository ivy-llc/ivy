# global
import jax
import jax.numpy as jnp
from typing import Optional

# local
import ivy
from ivy.functional.backends.jax import JaxArray


def bitwise_left_shift(
    x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None
) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.left_shift(x1, x2)


def add(x1: JaxArray, x2: JaxArray, *, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.add(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def bitwise_xor(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.bitwise_xor(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def exp(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.exp(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def expm1(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.expm1(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def bitwise_invert(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.bitwise_not(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def bitwise_and(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.bitwise_and(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def ceil(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        ret = x
    else:
        ret = jnp.ceil(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def floor(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        ret = x
    else:
        ret = jnp.floor(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def isfinite(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.isfinite(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def asin(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.arcsin(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def isinf(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.isinf(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def equal(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.equal(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def greater(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.greater(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def greater_equal(
    x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None
) -> JaxArray:
    ret = jnp.greater_equal(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def less_equal(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.less_equal(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def asinh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.arcsinh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sign(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.sign(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sqrt(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.sqrt(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def cosh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.cosh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def log10(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.log10(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def log(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.log(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def log2(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.log2(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def log1p(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.log1p(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def multiply(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.multiply(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def isnan(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.isnan(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def less(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.less(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def cos(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.cos(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def logical_xor(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.logical_xor(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def logical_or(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.logical_or(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def logical_and(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.logical_and(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def logical_not(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.logical_not(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def divide(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.divide(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def acos(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.arccos(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def acosh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.arccosh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sin(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.sin(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def negative(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.negative(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def not_equal(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.not_equal(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def tanh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.tanh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def floor_divide(
    x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None
) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.floor_divide(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def bitwise_or(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.bitwise_or(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sinh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.sinh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def positive(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.positive(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def square(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.square(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def pow(
    x1: jnp.ndarray, x2: jnp.ndarray, out: Optional[JaxArray] = None
) -> jnp.ndarray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    ret = jnp.power(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def remainder(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.remainder(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def round(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        ret = x
    else:
        ret = jnp.round(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def trunc(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        ret = x
    else:
        ret = jnp.trunc(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def abs(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.absolute(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def subtract(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    ret = jnp.subtract(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def logaddexp(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.logaddexp(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def bitwise_right_shift(
    x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None
) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.right_shift(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def tan(x: JaxArray) -> JaxArray:
    return jnp.tan(x)


def atan(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.arctan(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def atanh(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.arctanh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def atan2(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    ret = jnp.arctan2(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


# Extra #
# ------#


def minimum(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.minimum(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def maximum(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jnp.maximum(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def erf(x: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    ret = jax.scipy.special.erf(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret
