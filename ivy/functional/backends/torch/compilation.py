"""Collection of PyTorch compilation functions"""

# global
import torch

# local
import ivy


def compile(
    fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None
):
    if dynamic:
        return torch.jit.script(fn)
    if example_inputs is not None:
        example_inputs = ivy.to_native(example_inputs, nested=True)
    return torch.jit.trace(fn, example_inputs)
