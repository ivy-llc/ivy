import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def histogram(a, bins=10, range=None, density=None, weights=None):
    return ivy.histogram(a, bins, range, density, weights)
