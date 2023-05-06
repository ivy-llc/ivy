import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def to_tensor(data, dtype=None, place=None, stop_gradient=True):
    return ivy.array(data, dtype=dtype, device=place)
