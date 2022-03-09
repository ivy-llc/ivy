# global
import jax.numpy as jnp

# local
from ivy.functional.backends.jax import JaxArray


def bitwise_and(x1: JaxArray,
                x2: JaxArray)\
        -> JaxArray:
    return jnp.bitwise_and(x1, x2)


def ceil(x: JaxArray)\
        -> JaxArray:
    if 'int' in str(x.dtype):
        return x
    return jnp.ceil(x)


def isfinite(x: JaxArray)\
        -> JaxArray:
    return jnp.isfinite(x)


def asinh(x: JaxArray)\
        -> JaxArray:
    return jnp.arcsinh(x)


def sqrt(x: JaxArray) -> JaxArray:
    return jnp.sqrt(x)


def cosh(x: JaxArray)\
        -> JaxArray:
    return jnp.cosh(x)


def log2(x: JaxArray)\
        -> JaxArray:
    return jnp.log2(x)


def log1p(x: JaxArray)\
        -> JaxArray:
    return jnp.log1p(x)


def isnan(x: JaxArray)\
        -> JaxArray:
    return jnp.isnan(x)


def less(x1: JaxArray,x2:JaxArray)\
        -> JaxArray:
    return jnp.less(x1,x2)


def cos(x: JaxArray)\
        -> JaxArray:
    return jnp.cos(x)


def logical_not(x: JaxArray)\
        -> JaxArray:
    return jnp.logical_not(x)
