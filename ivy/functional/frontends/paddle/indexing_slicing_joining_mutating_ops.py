import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def reshape(input, shape):
    return ivy.reshape(input, shape)