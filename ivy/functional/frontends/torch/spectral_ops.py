import ivy
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def hann_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    device,
    layout=ivy.strides,
    required_grad=False
) -> ivy.Array:
    return ivy.hann_window(
        window_length, periodic, dtype, layout, device, required_grad
    )
