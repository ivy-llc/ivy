"""
Collection of Jax activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax
import jax.numpy as jnp

relu = lambda x: jnp.maximum(x, 0)
leaky_relu = lambda x, alpha=0.2: jnp.where(x > 0, x, x * alpha)
gelu = jax.nn.gelu
tanh = jnp.tanh
sigmoid = lambda x: 1 / (1 + jnp.exp(-x))


def softmax(x, axis=-1):
    exp_x = jnp.exp(x)
    return exp_x / jnp.sum(exp_x, axis, keepdims=True)


softplus = lambda x: jnp.log(jnp.exp(x) + 1)
