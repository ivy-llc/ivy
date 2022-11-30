import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def diagonal(input, *, offset=0, dim1=-2, dim2=-1):
    return torch_frontend.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)
