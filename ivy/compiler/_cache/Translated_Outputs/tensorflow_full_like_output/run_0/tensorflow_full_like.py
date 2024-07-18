from .tensorflow__helpers import tensorflow_asarray
from .tensorflow__helpers import tensorflow_full_like_1
from .tensorflow__helpers import tensorflow_to_scalar_1


def tensorflow_full_like(
    input,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    fill_value = tensorflow_to_scalar_1(tensorflow_asarray(fill_value))
    return tensorflow_full_like_1(input, fill_value, dtype=dtype, device=device)
