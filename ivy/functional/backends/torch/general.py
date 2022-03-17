"""
Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
import numpy as np
torch_scatter = None
import math as _math
import torch as _torch
from operator import mul
from torch.types import Number
from functools import reduce as _reduce
from typing import List, Dict, Optional, Union


# local
from ivy.functional.ivy.old import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.torch.device import dev_from_str, _callable_dev


def is_array(x, exclusive=False):
    if isinstance(x, _torch.Tensor):
        if exclusive and x.requires_grad:
            return False
        return True
    return False


def copy_array(x):
    return x.clone()


def array_equal(x0, x1):
    return _torch.equal(x0, x1)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray) or isinstance(x, (float, int, bool)):
        return x
    elif _torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise ValueError('Expected a pytroch tensor.')


def to_scalar(x) -> Union[float, int, bool]:
    if isinstance(x, (float, int)):
        return x
    return x.item()


def to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif _torch.is_tensor(x):
        return x.detach().cpu().tolist()
    raise ValueError('Expected a pytroch tensor.')