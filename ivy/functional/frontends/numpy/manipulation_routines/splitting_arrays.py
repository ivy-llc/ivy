# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting_special,
)


@handle_numpy_casting_special
@to_ivy_arrays_and_back
def split(ary, indices_or_sections = None, axis=0):
    return ivy.split(ary, indices_or_sections, axis)
