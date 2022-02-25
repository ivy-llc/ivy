# global
import torch
from torch import Tensor
from typing import Union, Tuple, Optional

# local
from ivy import dtype_from_str, default_dtype, dev_from_str, default_device


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


def eye(n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        dtype: Optional[Union[torch.dtype, str]] = None,
        device: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    dtype = dtype_from_str(default_dtype(dtype))
    device = dev_from_str(default_device(device))
    if n_cols is None:
        return torch.eye(n_rows, dtype=dtype, device=device)
    else:
        return torch.eye(n_rows, n_cols, dtype=dtype, device=device)
