import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def uniform(shape, minval=0, maxval=None, dtype=ivy.float32, seed=None, name=None):
    return ivy.random_uniform(
        low=minval, high=maxval, shape=shape, dtype=dtype, seed=seed
    )
