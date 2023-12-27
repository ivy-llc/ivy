import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"1.26.2 and below": ("int64",)}, "numpy")
@to_ivy_arrays_and_back
def bincount(x, /, weights=None, minlength=0):
    return ivy.bincount(x, weights=weights, minlength=minlength)


@with_supported_dtypes({"1.26.0 and below": ("int64",)}, "numpy")
@to_ivy_arrays_and_back
def histogram(a, bins=10, range=None, density=None, weights=None):
    return ivy.histogram(a, bins=bins, range=range, density=density, weights=weights)
