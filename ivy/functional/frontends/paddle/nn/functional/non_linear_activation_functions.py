# local
import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def tanh(input):
    return ivy.tanh(input)