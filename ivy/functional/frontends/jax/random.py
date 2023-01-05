import ivy

from functools import partial

try:
    from jax._src.prng import PRNGKeyArray as _PRNGKeyArray
except ImportError:
    from types import SimpleNamespace

    random = SimpleNamespace


def PRNGKey(seed: int) -> ivy.Array:
    ivy.seed(seed_value=seed)
    return random().PRNG(seed)
