# global
import ivy
from ivy.functional.frontends.scipy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def tril(m, k=0):
    return ivy.tril(m, k=k)


@to_ivy_arrays_and_back
def triu(m, k=0):
    return ivy.triu(m, k=k)
