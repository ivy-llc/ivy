"""
Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch


def random_uniform(low, high, size, dev='cpu'):
    rand_range = high - low
    return _torch.rand(*size).to(dev) * rand_range + low


randint = lambda low, high, size, dev='cpu': _torch.randint(low, high, size).to(dev)


def seed(seed_value):
    _torch.manual_seed(seed_value)
    _torch.cuda.manual_seed(seed_value)
    return


def shuffle(x):
    batch_size = x.shape[0]
    return x[_torch.randperm(batch_size)]
