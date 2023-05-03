import ivy.functional.frontends.numpy as np_frontend
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def tril(m, k=0):
    return np_frontend.tril(m, k=k)


@to_ivy_arrays_and_back
def triu(m, k=0):
    return np_frontend.triu(m, k=k)
