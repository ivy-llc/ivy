"""
Collection of Jax math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax as _jax
import jax.numpy as _jnp


tan = _jnp.tan
asin = _jnp.arcsin
acos = _jnp.arccos
atan = _jnp.arctan
atan2 = _jnp.arctan2
cosh = _jnp.cosh
atanh = _jnp.arctanh
log = _jnp.log
exp = _jnp.exp
erf = _jax.scipy.special.erf
