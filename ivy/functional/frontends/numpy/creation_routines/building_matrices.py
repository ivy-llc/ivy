import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def tril(m, k=0):
    return ivy.tril(m, k=k)


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)
