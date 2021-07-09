"""
Collection of Jax math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax as _jax
import jax.numpy as _jnp

sin = _jnp.sin
cos = _jnp.cos
tan = _jnp.tan
asin = _jnp.arcsin
acos = _jnp.arccos
atan = _jnp.arctan
atan2 = _jnp.arctan2
sinh = _jnp.sinh
cosh = _jnp.cosh
tanh = _jnp.tanh
asinh = _jnp.arcsinh
acosh = _jnp.arccosh
atanh = _jnp.arctanh
log = _jnp.log
exp = _jnp.exp
erf = _jax.scipy.special.erf
