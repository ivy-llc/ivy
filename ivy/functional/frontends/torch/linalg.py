# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
def diagonal(input, *, offset=0, dim1=-2, dim2=-1):
    return torch_frontend.diagonal(input, offset=offset, dim1=dim1, dim2=dim2)


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


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def cross(input, other, *, dim=-1, out=None):
    return torch_frontend.cross(input, other, dim, out=out)
