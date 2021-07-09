"""
Collection of Jax activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax as _jax
import jax.numpy as _jnp

relu = lambda x: _jnp.maximum(x, 0)
leaky_relu = lambda x, alpha=0.2: _jnp.where(x > 0, x, x * alpha)
gelu = _jax.nn.gelu
tanh = _jnp.tanh
sigmoid = lambda x: 1 / (1 + _jnp.exp(-x))


def softmax(x, axis=-1):
    exp_x = _jnp.exp(x)
    return exp_x / _jnp.sum(exp_x, axis, keepdims=True)


softplus = lambda x: _jnp.log(_jnp.exp(x) + 1)
