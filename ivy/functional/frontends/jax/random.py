# global
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def bernoulli(key, p, shape):
    return ivy.bernoulli(p, shape=shape, seed=int(key[-1]))
