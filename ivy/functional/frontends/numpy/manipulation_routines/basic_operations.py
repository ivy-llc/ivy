# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    inputs_to_ivy_arrays,
    handle_numpy_casting,
)


@inputs_to_ivy_arrays
@handle_numpy_casting
def copyto(dst, src, /, *, casting="same_kind", where=True):
    if ivy.is_array(where):
        ivy.where(where, src, dst, out=dst)


@inputs_to_ivy_arrays
def shape(array, /):
    return ivy.shape(array)
