import ivy
from ivy.functional.frontends.mxnet.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def randint(low, high=None, size=None, dtype=None, device=None, out=None):
    return ivy.randint(low, high=high, shape=size, device=device, dtype=dtype, out=out)


@to_ivy_arrays_and_back
def uniform(low=0.0, high=1.0, size=None, dtype=None, device=None, out=None):
    return ivy.random_uniform(low=low, high=high, shape=size, device=device, dtype=dtype, out=out)
