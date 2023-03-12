# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def tile(A, reps):
    return ivy.tile(A, reps)


@to_ivy_arrays_and_back
def repeat(a, repeats, axis=None):
    return ivy.repeat(a, repeats, axis=axis)
