# global
import ivy
from .tensor import Tensor
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def to_tensor(data, dtype=None, place=None, stop_gradient=True):
    array = ivy.array(data, dtype=dtype, device=place)
    return Tensor(array, dtype=dtype, place=place)
