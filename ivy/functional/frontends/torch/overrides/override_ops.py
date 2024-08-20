# global
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def is_tensor_like(inp):
    return ivy.is_array(inp) 