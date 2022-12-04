# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def diagonal(input, *, offset=0, dim1=-2, dim2=-1):
    return torch_frontend.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)


@to_ivy_arrays_and_back
def inv(input, *, out=None):
    return ivy.inv(input, out=out)


@to_ivy_arrays_and_back
def det(input, *, out=None):
    return ivy.det(input, out=out)


@to_ivy_arrays_and_back
def slogdet(input, *, out=None):
    return ivy.slogdet(input, out=out)


@to_ivy_arrays_and_back
def matrix_power(input, n, *, out=None):
    return ivy.matrix_power(input, n, out=out)
