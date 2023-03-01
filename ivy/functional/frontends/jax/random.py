# global
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


# added the wrapper to virtually all functions in the frontends
@to_ivy_arrays_and_back
def cauchy(key, shape, dtype="float"):
    return ivy.cauchy(key, shape, dtype)
