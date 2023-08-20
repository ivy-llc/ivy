import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16")},
    "torch",
)
def hamming_window(
    window_length,
    alpha=0.54,
    beta=0.46,
    periodic=True,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False
):
    return ivy.hamming_window(
        window_length, alpha=alpha, beta=beta, periodic=periodic, dtype=dtype
    )
