# global
import jax
import jax.numpy as jnp
from typing import Optional

# local
import ivy
from ivy.functional.backends.jax import JaxArray


def bitwise_left_shift(x1: JaxArray,
                       x2: JaxArray,
                       out: Optional[JaxArray] = None)\
        -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.left_shift(x1, x2)


def add(x1: JaxArray,
        x2: JaxArray,
        out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.add(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def bitwise_xor(x1: JaxArray,
                x2: JaxArray,
                out: Optional[JaxArray] = None)\
        -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.bitwise_xor(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def exp(x: JaxArray,
        out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.exp(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def expm1(x: JaxArray,
          out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.expm1(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def bitwise_invert(x: JaxArray,
                   out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.bitwise_not(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def bitwise_and(x1: JaxArray,
                x2: JaxArray,
                out: Optional[JaxArray] = None)\
        -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.bitwise_and(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def ceil(x: JaxArray,
         out: Optional[JaxArray] = None)\
        -> JaxArray:
    if 'int' in str(x.dtype):
        ret = x
    else:
        ret = jnp.ceil(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def floor(x: JaxArray,
          out: Optional[JaxArray] = None)\
        -> JaxArray:
    if 'int' in str(x.dtype):
        ret = x
    else:
        ret = jnp.floor(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def isfinite(x: JaxArray)\
        -> JaxArray:
    return jnp.isfinite(x)


def asin(x: JaxArray,
         out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.arcsin(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def isinf(x: JaxArray)\
        -> JaxArray:
    return jnp.isinf(x)


def equal(x1: JaxArray,
          x2: JaxArray,
          out: Optional[JaxArray] = None)\
        -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    ret = jnp.equal(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def greater(x1: JaxArray, x2: JaxArray) \
        -> JaxArray:
    return jnp.greater(x1, x2)


def greater_equal(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return jnp.greater_equal(x1, x2)


def less_equal(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    return x1 <= x2


def asinh(x: JaxArray)\
        -> JaxArray:
    return jnp.arcsinh(x)

def sign(x: JaxArray)\
        -> JaxArray:
    return jnp.sign(x)

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


def multiply(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.multiply(x1, x2)


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
  
  
def divide(x1: JaxArray,
           x2: JaxArray)\
        -> JaxArray:
    return jnp.divide(x1, x2)  


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
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.not_equal(x1, x2)


def tanh(x: JaxArray)\
        -> JaxArray:
    return jnp.tanh(x)


def floor_divide(x1: JaxArray, x2: JaxArray)\
                -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.floor_divide(x1, x2)


def bitwise_or(x1: JaxArray, x2: JaxArray) -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
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


def pow(x1: jnp.ndarray, x2: jnp.ndarray)\
        -> jnp.ndarray:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    return jnp.power(x1, x2)


def remainder(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.remainder(x1, x2)


def round(x: JaxArray)\
        -> JaxArray:
    if 'int' in str(x.dtype):
        return x
    return jnp.round(x)


def trunc(x: JaxArray)\
        -> JaxArray:
    if 'int' in str(x.dtype):
        return x
    return jnp.trunc(x)

  
def abs(x: JaxArray,
        out: Optional[JaxArray] = None)\
        -> JaxArray:
    ret = jnp.absolute(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def subtract(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    return jnp.subtract(x1, x2)


def logaddexp(x1: JaxArray, x2: JaxArray) -> JaxArray:
    return jnp.logaddexp(x1, x2)


def bitwise_right_shift(x1: JaxArray, x2: JaxArray)\
        -> JaxArray:
    if isinstance(x2, int):
        x2 = jnp.asarray(x2, dtype=x1.dtype)
    return jnp.right_shift(x1, x2)


tan = jnp.tan


def atan(x: JaxArray)\
        -> JaxArray:
    return jnp.arctan(x)



def atanh(x: JaxArray)\
        -> JaxArray:
    return jnp.arctanh(x)






def atan2(x1: JaxArray, x2: JaxArray) -> JaxArray:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = jnp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    return jnp.arctan2(x1, x2)



cosh = jnp.cosh
log = jnp.log
exp = jnp.exp

# Extra #
# ------#

minimum = jnp.minimum
maximum = jnp.maximum
erf = jax.scipy.special.erf
