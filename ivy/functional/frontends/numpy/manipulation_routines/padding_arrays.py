# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def pad(array, pad_width, mode="constant", **kwargs):
    return ivy.pad(array, pad_width, mode=mode, **kwargs)
