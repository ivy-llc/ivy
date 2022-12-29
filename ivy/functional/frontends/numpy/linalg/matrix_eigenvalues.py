# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def eigvalsh(x, /):
    return ivy.eigvalsh(x)
