"""
Collection of Jax random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax as _jax
import jax.numpy as _jnp

# local
from ivy.jax.core.general import to_dev

RNG = _jax.random.PRNGKey(0)


def random_uniform(low=0.0, high=1.0, shape=None, dev_str='cpu'):
    global RNG
    RNG, rng_input = _jax.random.split(RNG)
    return to_dev(_jax.random.uniform(rng_input, shape if shape else (), minval=low, maxval=high), dev_str)


def multinomial(probs, num_samples):
    global RNG
    RNG, rng_input = _jax.random.split(RNG)
    orig_probs_shape = list(probs.shape)
    num_classes = orig_probs_shape[-1]
    probs_flat = _jnp.reshape(probs, (-1, orig_probs_shape[-1]))
    probs_flat = probs_flat / _jnp.sum(probs_flat, -1, keepdims=True)
    probs_stack = _jnp.split(probs_flat, probs_flat.shape[0])
    samples_stack = [_jax.random.choice(rng_input, num_classes, (num_samples,), p=prob[0]) for prob in probs_stack]
    samples_flat = _jnp.stack(samples_stack)
    return _jnp.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples])


def randint(low, high, shape, dev_str='cpu'):
    global RNG
    RNG, rng_input = _jax.random.split(RNG)
    return to_dev(_jax.random.randint(rng_input, shape, low, high), dev_str)


def seed(seed_value=0):
    global RNG
    RNG = _jax.random.PRNGKey(seed_value)
    return


def shuffle(x):
    global RNG
    RNG, rng_input = _jax.random.split(RNG)
    return _jax.random.shuffle(rng_input, x)
