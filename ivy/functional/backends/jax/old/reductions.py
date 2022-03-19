"""
Collection of Jax reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax.numpy as _jnp


def einsum(equation, *operands):
    return _jnp.einsum(equation, *operands)

