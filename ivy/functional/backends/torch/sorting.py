# global
import torch


def argsort(x: torch.Tensor,
            axis: int = -1,
            descending: bool = False,
            stable: bool = True) \
        -> torch.Tensor:
    return torch.argsort(x, dim=axis, descending=descending)


def sort(x: torch.Tensor,
         axis: int = -1,
         descending: bool = False,
         stable: bool = True) -> torch.Tensor:
    sorted_tensor, _ = torch.sort(x, dim=axis, descending=descending)
    return sorted_tensor
