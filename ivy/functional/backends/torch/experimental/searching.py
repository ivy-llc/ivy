# global
from typing import Optional, Tuple
import torch


def unravel_index(
    indices: torch.Tensor,
    shape: Tuple[int],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    temp = indices.to(torch.int32)
    output = []
    for dim in reversed(shape):
        output.append(temp % dim)
        temp = temp // dim
    return tuple(reversed(output))


unravel_index.support_native_out = False
