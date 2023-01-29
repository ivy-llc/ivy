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
    keys: Union[torch.Tensor, list],
    /,
    *,
    axis: int = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pass  # TODO: implement lexsort for torch


lexsort_support_native_out = False
