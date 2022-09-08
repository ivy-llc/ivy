"""Collection of PyTorch compilation functions"""

# global
from typing import Callable, Any, Union, Sequence, Iterable, Optional
import torch

# local
import ivy


def compile(
    fn: Callable,
    /,
    *,
    dynamic: bool = True,
    example_inputs: Optional[Union[Any, Sequence[Any]]] = None,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
) -> Callable:
    if dynamic:
        return torch.jit.script(fn)
    if example_inputs is not None:
        example_inputs = ivy.to_native(example_inputs, nested=True)
    return torch.jit.trace(fn, example_inputs)
