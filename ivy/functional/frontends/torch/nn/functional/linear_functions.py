# local
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def linear(input, weight, bias=None):
    return ivy.linear(input, weight, bias=bias)
