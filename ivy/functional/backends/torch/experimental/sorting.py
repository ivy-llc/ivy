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
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x_unq, inv = torch.unique(x.flip(0), dim=axis, sorted=True, return_inverse=True)
    return torch.argsort(inv)


lexsort_support_native_out = True
