# global
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def is_tensor_like(inp):
    return ivy.is_array(inp)
