"""Collection of Jax compilation functions."""

# global
import jax as jax


def compile(
    fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None
):
    return jax.jit(fn, static_argnums=static_argnums, static_argnames=static_argnames)
