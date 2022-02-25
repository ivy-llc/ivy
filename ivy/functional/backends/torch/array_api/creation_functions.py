# global
from typing import Union, Tuple
import torch
from torch import Tensor

# local
from ivy import dtype_from_str, default_dtype, dev_from_str, default_device


# noinspection PyShadowingName
def zeros(shape: Union[int, Tuple[int, ...]],
          dtype: torch.dtype = None,
          device: torch.device = None) \
        -> Tensor:
    return torch.zeros(shape, dtype=dtype_from_str(default_dtype(dtype)), device=dev_from_str(default_device(device)))