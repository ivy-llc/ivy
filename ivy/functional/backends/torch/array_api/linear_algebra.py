#global
import torch

def det(A:torch.Tensor) \
    -> torch.Tensor:
    return torch.linalg.det(A)
