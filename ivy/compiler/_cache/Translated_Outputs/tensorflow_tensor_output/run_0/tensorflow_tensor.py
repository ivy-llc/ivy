from .tensorflow__helpers import tensorflow_asarray


def tensorflow_tensor(
    data, *, dtype=None, device=None, requires_grad=False, pin_memory=False
):
    return tensorflow_asarray(data, dtype=dtype, device=device)
