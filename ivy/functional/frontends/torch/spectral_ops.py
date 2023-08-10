import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "float32", "float64", "complex")},
    "torch",
)
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
