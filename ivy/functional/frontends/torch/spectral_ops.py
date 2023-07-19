import ivy
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"2.0.1 and below": "int"}, "torch")
@to_ivy_arrays_and_back
def hann_window(
    window_length, periodic=True, *, dtype=None, requires_grad=False, out=None
):
    return ivy.hann_window(
        window_length, periodic, dtype, requires_grad=requires_grad, out=out
    )
