import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def hamming_window(
    window_length,
    periodic=True,
    alpha=0.54,
    beta=0.46,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False
):
    return ivy.hamming_window(
        window_length, periodic=periodic, alpha=alpha, beta=beta, dtype=dtype
    )
