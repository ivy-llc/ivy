import torch


def unique_values(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.unique(x)
