import torch
from typing import Optional, Union

def argmax(
    x: torch.Tensor,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    keepdims: Optional[bool] = False
) -> torch.Tensor:
    return torch.argmax(x,dim=axis,out=out,keepdim=keepdims)