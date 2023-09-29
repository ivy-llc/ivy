import ivy
from ivy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def Flatten(x):
    flattened = ivy.flatten(x, copy=None, start_dim=0, end_dim=-1, order="C", out=None)
    return flattened
