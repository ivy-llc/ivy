import torch
from typing import Optional, Union

def argmax(
    x: torch.Tensor,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    keepdims: Optional[bool] = False
) -> torch.Tensor:
    x = torch.tensor(x)
    ret = torch.argmax(x,dim=axis,out=out,keepdim=keepdims)
    return ret


def argmin(
    x: torch.Tensor,
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False)\
    -> torch.Tensor:

    x = torch.tensor(x)
    ret = torch.argmin(x,axis=axis, keepdim=keepdims)

    return ret


def where(condition, x1, x2):
    return torch.where(condition.type(torch.bool), x1, x2)

