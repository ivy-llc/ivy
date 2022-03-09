# global
import torch
from torch import Tensor
from typing import Union, Tuple, Optional, Dict


# local
from ivy import dtype_from_str, default_dtype, dev_from_str, default_device
from ivy.functional.backends.torch.core.device import _callable_dev


def zeros(shape: Union[int, Tuple[int]],
          dtype: Optional[torch.dtype] = None,
          device: Optional[torch.device] = None) \
        -> Tensor:
    return torch.zeros(shape, dtype=dtype_from_str(default_dtype(dtype)), device=dev_from_str(default_device(device)))


def ones(shape: Union[int, Tuple[int]],
         dtype: Optional[torch.dtype] = None,
         device: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    dtype_val: torch.dtype = dtype_from_str(dtype)
    dev = default_device(device)
    return torch.ones(shape, dtype=dtype_val, device=dev_from_str(dev))


# noinspection DuplicatedCode
def full_like(x: torch.Tensor, /,
              fill_value: Union[int, float],
              dtype: Optional[torch.dtype] = None,
              device: Optional[Union[torch.device, str, None]] = None) \
        -> torch.Tensor:
    if device is None:
        device = _callable_dev(x)
    if dtype is not None:
        return torch.full_like(x, fill_value, dtype=dtype, device=dev_from_str(device))

    return torch.full_like(x, fill_value, device=dev_from_str(device))


def tril(x: torch.Tensor,
         k: int = 0) \
         -> torch.Tensor:
    return torch.tril(x, diagonal=k)

