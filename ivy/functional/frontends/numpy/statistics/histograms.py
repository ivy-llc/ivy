import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"1.25.2 and below": ("int64",)}, "numpy")
@to_ivy_arrays_and_back
def bincount(x, /, weights=None, minlength=0):
    return ivy.bincount(x, weights=weights, minlength=minlength)
