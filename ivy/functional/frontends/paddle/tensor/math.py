# global
import ivy
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)
import ivy.functional.frontends.paddle as paddle_frontend


@to_ivy_arrays_and_back
def sin(x, name=None):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def cos(x, name=None):
    return ivy.cos(x)


@to_ivy_arrays_and_back
def tan(x, name=None):
    return paddle_frontend.tan(x)
