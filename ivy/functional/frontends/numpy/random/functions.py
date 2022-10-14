# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def random(size=None):
    return ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
