# global
import jax
import typing
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def expm1(x: JaxArray)\
        -> JaxArray:
    return jnp.expm1(x)


def bitwise_invert(x: JaxArray)\
        -> JaxArray:
    return jnp.bitwise_not(x)


def bitwise_and(x1: JaxArray,
                x2: JaxArray)\
        -> JaxArray:
    return jnp.bitwise_and(x1, x2)


def ceil(x: JaxArray)\
        -> JaxArray:
    if 'int' in str(x.dtype):
        return x
    return jnp.ceil(x)


def floor(x: JaxArray)\
        -> JaxArray:
    if 'int' in str(x.dtype):
        return x
    return jnp.floor(x)


def isfinite(x: JaxArray)\
        -> JaxArray:
    return jnp.isfinite(x)


def asin(x: JaxArray)\
        -> JaxArray:
    return jnp.arcsin(x)


def isinf(x: JaxArray)\
        -> JaxArray:
    return jnp.isinf(x)


def _cast_for_binary_op(x1: JaxArray, x2: JaxArray)\
        -> typing.Tuple[typing.Union[JaxArray, int, float, bool], typing.Union[JaxArray, int, float, bool]]:
    x1_bits = ivy.functional.backends.jax.old.general.dtype_bits(x1.dtype)
    if isinstance(x2, (int, float, bool)):
        return x1, x2
    x2_bits = ivy.functional.backends.jax.old.general.dtype_bits(x2.dtype)
    if x1_bits > x2_bits:
        x2 = x2.astype(x1.dtype)
    elif x2_bits > x1_bits:
        x1 = x1.astype(x2.dtype)
    return x1, x2


def equal(x1: JaxArray, x2: JaxArray) -> JaxArray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return x1 == x2


def greater_equal(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return jnp.greater_equal(x1, x2)


def less_equal(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return x1 <= x2


def asinh(x: JaxArray)\
        -> JaxArray:
    return jnp.arcsinh(x)


def sqrt(x: JaxArray)\
        -> JaxArray:
    return jnp.sqrt(x)


def cosh(x: JaxArray)\
        -> JaxArray:
    return jnp.cosh(x)


def log10(x: JaxArray)\
        -> JaxArray:
    return jnp.log10(x)


def log(x: JaxArray)\
        -> JaxArray:
    return jnp.log(x)


def log2(x: JaxArray)\
        -> JaxArray:
    return jnp.log2(x)


def log1p(x: JaxArray)\
        -> JaxArray:
    return jnp.log1p(x)


def isnan(x: JaxArray)\
        -> JaxArray:
    return jnp.isnan(x)


def less(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return jnp.less(x1, x2)


def cos(x: JaxArray)\
        -> JaxArray:
    return jnp.cos(x)


def logical_xor(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return jnp.logical_xor(x1, x2)


def logical_or(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return jnp.logical_or(x1, x2)


def logical_and(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return jnp.logical_and(x1, x2)


def logical_not(x: JaxArray)\
        -> JaxArray:
    return jnp.logical_not(x)


def acos(x: JaxArray)\
        -> JaxArray:
    return jnp.arccos(x)


def acosh(x: JaxArray)\
        -> JaxArray:
    return jnp.arccosh(x)


def sin(x: JaxArray)\
        -> JaxArray:
    return jnp.sin(x)


def negative(x: JaxArray) -> JaxArray:
    return jnp.negative(x)


def not_equal(x1: JaxArray, x2: JaxArray) \
        -> JaxArray:
    return jnp.not_equal(x1, x2)


def tanh(x: JaxArray)\
        -> JaxArray:
    return jnp.tanh(x)


def bitwise_or(x1: JaxArray, x2: JaxArray) -> JaxArray:
    if isinstance(x1,int):
        if x1 > 9223372036854775807:
           x1 = jnp.array(x1,dtype='uint64')

    if isinstance(x2,int):
        if x2 > 9223372036854775807:
           x2 = jnp.array(x2,dtype='uint64')

    return jnp.bitwise_or(x1, x2)


def sinh(x: JaxArray)\
        -> JaxArray:
    return jnp.sinh(x)


def positive(x: JaxArray)\
        -> JaxArray:
    return jnp.positive(x)


def square(x: JaxArray)\
        -> JaxArray:
    return jnp.square(x)


def remainder(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return jnp.remainder(x1, x2)


def round(x: JaxArray)\
        -> JaxArray:
    if 'int' in str(x.dtype):
        return x
    return jnp.round(x)


def abs(x: JaxArray)\
        -> JaxArray:
    return jnp.absolute(x)


def subtract(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    return jnp.subtract(x1, x2)


def logaddexp(x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.logaddexp(x1, x2)


tan = jnp.tan


def atan(x: JaxArray)\
        -> JaxArray:
    return jnp.arctan(x)


def atan2(x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.arctan2(x1, x2)


cosh = jnp.cosh
atanh = jnp.arctanh
log = jnp.log
exp = jnp.exp

# Extra #
# ------#


erf = jax.scipy.special.erf
