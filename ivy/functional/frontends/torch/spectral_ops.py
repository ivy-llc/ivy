import ivy
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"2.0.1 and below": "int"}, "torch")
@to_ivy_arrays_and_back
def hann_window(size, *, periodic=True, dtype=None, out=None) -> ivy.Array:
    return ivy.hann_window(size, periodic, dtype, out)
