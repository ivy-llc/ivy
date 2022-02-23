# global
import torch
from typing import Tuple, Optional, Union

# local
from ivy.functional.ivy.core import default_device
from ivy.functional.backends.torch import dev_from_str, dtype_from_str

# noinspection PyShadowingNames
def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[torch.dtype] = 'float32',
         device: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    dtype_val: torch.dtype = dtype_from_str(dtype)
    dev = default_device(device)
    return torch.ones(shape, dtype=dtype_val, device=dev_from_str(dev))