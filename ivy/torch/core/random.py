"""
Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch
from typing import Optional, List


def random_uniform(low: float = 0.0, high: float = 1.0, shape: Optional[List[int]] = None, dev: str = 'cpu'):
    rand_range = high - low
    return torch.rand(shape).to(dev.replace('gpu', 'cuda')) * rand_range + low


def randint(low: int, high: int, shape: List[int], dev: torch.device = 'cpu'):
    return torch.randint(low, high, shape).to(dev)


def seed(seed_value: int) -> None:
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return


def shuffle(x):
    batch_size = x.shape[0]
    return x[torch.randperm(batch_size)]
