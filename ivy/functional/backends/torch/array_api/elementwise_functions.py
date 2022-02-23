# global
import torch

def isfinite(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.isfinite(x)

def less(x1: torch.Tensor,x2: torch.Tensor):
    if hasattr(x1,'dtype') and hasattr(x2,'dtype'):
        promoted_type = torch.promote_types(x1.dtype,x2.dtype)
        x1 = x1.to(promoted_type)
        x2 = x2.to(promoted_type)
    
    return torch.lt(x1,x2)