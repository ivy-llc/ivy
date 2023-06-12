# global


# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def randint(low=0, high=None, shape=[1], dtype=None, name=None):
    return ivy.randint(low, high, shape=shape, dtype=dtype)
