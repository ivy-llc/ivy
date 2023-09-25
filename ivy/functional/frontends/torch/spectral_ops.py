import ivy
from ivy import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def blackman_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False
):
    return ivy.blackman_window(window_length, periodic=periodic, dtype=dtype)
