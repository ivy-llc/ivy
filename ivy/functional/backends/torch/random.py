"""
Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature.
"""

# global

import torch
from typing import Optional, List

# local
import ivy
from ivy.functional.ivy.device import default_device


# Extra #
# ------#

def random_uniform(low: float = 0.0, high: float = 1.0, shape: Optional[List[int]] = None, dev: ivy.Device = None):
    rand_range = high - low
    if shape is None:
        true_shape: List[int] = []
    else:
        true_shape: List[int] = shape
    return torch.rand(true_shape, device=default_device(dev)) * rand_range + low


def random_normal(mean: float = 0.0, std: float = 1.0, shape: Optional[List[int]] = None, dev: ivy.Device = None):
    if shape is None:
        true_shape: List[int] = []
    else:
        true_shape: List[int] = shape
    mean = mean.item() if isinstance(mean, torch.Tensor) else mean
    std = std.item() if isinstance(std, torch.Tensor) else std
    return torch.normal(mean, std, true_shape, device=default_device(dev).replace('gpu', 'cuda'))


def multinomial(population_size: int, num_samples: int, batch_size: int, probs: Optional[torch.Tensor] = None,
                replace: bool = True, dev: ivy.Device = None):
    if probs is None:
        probs = torch.ones((batch_size, population_size,)) / population_size
    return torch.multinomial(probs, num_samples, replace).to(default_device(dev))


def randint(low: int, high: int, shape: List[int], dev: ivy.Device = None):
    return torch.randint(low, high, shape, device=default_device(dev))


def seed(seed_value: int = 0) -> None:
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return


def shuffle(x):
    batch_size = x.shape[0]
    return torch.index_select(x, 0, torch.randperm(batch_size))
