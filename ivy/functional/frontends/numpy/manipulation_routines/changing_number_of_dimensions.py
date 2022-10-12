# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.numpy.mathematical_functions import prod


# squeeze
@to_ivy_arrays_and_back
def squeeze(
    a,
    axis=None,
):
    return ivy.squeeze(a, axis)


# expand_dims
@to_ivy_arrays_and_back
def expand_dims(
    a,
    axis,
):
    return ivy.expand_dims(a, axis=axis)


# flatten
@to_ivy_arrays_and_back
def flatten(
    a,
    /,
    order="C",
):
    a = ivy.reshape(a, (1, prod(a.shape())))
    return ivy.squeeze(a)
