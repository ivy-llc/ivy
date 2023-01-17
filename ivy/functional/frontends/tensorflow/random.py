import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def uniform(shape, minval=0., maxval=1., dtype=ivy.float32, seed=None):
    return ivy.random_uniform(shape=shape,
                              low=minval,
                              high=maxval,
                              dtype=dtype,
                              seed=seed)
