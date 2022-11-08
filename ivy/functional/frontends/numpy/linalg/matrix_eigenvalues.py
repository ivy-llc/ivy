import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


# eig
@to_ivy_arrays_and_back
def eig(a):
    return ivy.eig(a)
