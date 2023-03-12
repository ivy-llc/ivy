import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def stop_gradient(x):
    return ivy.stop_gradient(x)
