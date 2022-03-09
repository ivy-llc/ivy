#global
import torch

def det(x:torch.Tensor) \
    -> torch.Tensor:
    return torch.linalg.det(x)
