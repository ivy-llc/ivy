from .tensorflow__helpers import tensorflow_zeros_like_1


def tensorflow_zeros_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    ret = tensorflow_zeros_like_1(input, dtype=dtype, device=device)
    return ret
