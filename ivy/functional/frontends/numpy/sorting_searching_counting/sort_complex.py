import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
)

@handle_numpy_casting
@to_ivy_arrays_and_back
def sort_complex(array):
    if len(array) == 1:
        return array
    else:
        return ivy.sort(array)
