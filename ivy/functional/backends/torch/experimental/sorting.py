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
    x_unq = torch.unique(x.flip(0), dim=axis, sorted=True, return_inverse=True)
    inv = []
    for i in range(0, x.size()[1]):
        for j in range(0, x.size()[1]):
            if (x.flip(0)[:, j] == x_unq[:, i]).all():
                inv.append(j)
    return inv


lexsort_support_native_out = True
