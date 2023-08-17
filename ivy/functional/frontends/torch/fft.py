import ivy
from ivy import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def rfftn(input, s=None, dim=None, norm=None, *, out=None):
    if dim is None:
        dim = [-1]
    if s is None:
        s = [input.size]
    return ivy.rfftn(input, s=s, axes=dim, norm=norm)
