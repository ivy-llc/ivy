# global
import torch
from typing import Optional, Union


# msort
def msort(
    a: Union[torch.Tensor, list, tuple], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.msort(a, out=out)


msort_support_native_out = True


# lexsort
def lexsort(
    x: Union[torch.Tensor, list, tuple],
    /,
    *,
    axis: int = -1,
) -> torch.Tensor:
    inverse_indices = torch.unique(x.flip(0), sorted=True, return_inverse=True, return_counts=False, dim=axis)[1]
    return torch.argsort(inverse_indices, dim=axis)


lexsort_support_native_out = True
