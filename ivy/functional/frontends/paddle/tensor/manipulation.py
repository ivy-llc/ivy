# global
import ivy
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def reshape(x, shape):
    return ivy.reshape(x, shape)
