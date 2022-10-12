# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def reshape(x, /, newshape, order="C"):
    return ivy.reshape(x, shape=newshape)
