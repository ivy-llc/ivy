import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def sin(x, name=None):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def cos(x, name=None):
    return ivy.cos(x)
