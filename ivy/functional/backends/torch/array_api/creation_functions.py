# global
import torch
from typing import Tuple, Optional, Union

# local
import ivy


# noinspection PyShadowingNames
def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[torch.dtype] = None,
         device: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    dtype_val: torch.dtype = ivy.dtype_from_str(dtype)
    dev = ivy.default_device(device)
    return torch.ones(shape, dtype=dtype_val, device=ivy.dev_from_str(dev))
