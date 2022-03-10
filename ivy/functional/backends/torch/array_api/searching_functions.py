# global
import torch
from typing import Optional

def argmin(
    x: torch.Tensor,
    dim: Optional[int] = None,
    keepdims: Optional[bool] = False
    ) -> torch.Tensor:

    x = torch.tensor(x)
    ret = torch.argmin(x,dim=dim, keepdim=keepdims)

    return ret
