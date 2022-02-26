#global
import torch

#local
from typing import Optional

def det(A:torch.Tensor,
        out:Optional[torch.Tensor]=None) \
    -> torch.Tensor:
    return torch.linalg.det(A,out)
