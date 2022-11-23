# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import inputs_to_ivy_arrays


@inputs_to_ivy_arrays
def shape(array, /):
    return ivy.shape(array)
