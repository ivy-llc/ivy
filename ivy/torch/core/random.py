"""
Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch
from typing import Optional, List

# local
from ivy.core.device import default_device


def random_uniform(low: float = 0.0, high: float = 1.0, shape: Optional[List[int]] = None, dev_str: str = None):
    rand_range = high - low
    if shape is None:
        true_shape: List[int] = []
    else:
        true_shape: List[int] = shape
    return _torch.rand(true_shape, device=default_device(dev_str).replace('gpu', 'cuda')) * rand_range + low


def random_normal(mean: float = 0.0, std: float = 1.0, shape: Optional[List[int]] = None, dev_str: str = None):
    if shape is None:
        true_shape: List[int] = []
    else:
        true_shape: List[int] = shape
    mean = mean.item() if isinstance(mean, _torch.Tensor) else mean
    std = std.item() if isinstance(std, _torch.Tensor) else std
    return _torch.normal(mean, std, true_shape, device=default_device(dev_str).replace('gpu', 'cuda'))


def multinomial(population_size: int, num_samples: int, batch_size: int, probs: Optional[_torch.Tensor] = None,
                replace: bool = True, dev_str: str = None):
    if probs is None:
        probs = _torch.ones((batch_size, population_size,)) / population_size
    return _torch.multinomial(probs, num_samples, replace).to(default_device(dev_str).replace('gpu', 'cuda'))


def randint(low: int, high: int, shape: List[int], dev_str: str = None):
    return _torch.randint(low, high, shape, device=default_device(dev_str).replace('gpu', 'cuda'))


def seed(seed_value: int = 0) -> None:
    _torch.manual_seed(seed_value)
    _torch.cuda.manual_seed(seed_value)
    return


def shuffle(x):
    batch_size = x.shape[0]
    return _torch.index_select(x, 0, _torch.randperm(batch_size))
