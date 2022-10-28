import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    return ivy.multinomial(input, num_samples, replace=replacement, out=out)
