from kornia.core import tensor, pad, stack, Module, Parameter


def kornia_func():
    arr = tensor([1, 2, 3])
    arr = pad(arr, (1, 1), mode="constant", value=0)
    arr = stack([arr, arr])
    param = Parameter(arr)
    mod = Module()
